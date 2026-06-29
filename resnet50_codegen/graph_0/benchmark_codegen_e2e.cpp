#include <turbojpeg.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <mutex>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <tt-metalium/distributed.hpp>

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) ||          \
    defined(_M_IX86)
#include <immintrin.h>
#endif

#define TTNN_STANDALONE_DISABLE_MAIN
#ifndef TTNN_CODEGEN_STANDALONE_CPP
#define TTNN_CODEGEN_STANDALONE_CPP "ttnn-standalone.cpp"
#endif
#include TTNN_CODEGEN_STANDALONE_CPP

namespace {

constexpr int kResizeShorterSide = 256;
constexpr int kCropSize = 224;
constexpr int kChannels = 3;
constexpr int kInputTensorIndex = 27;
constexpr float kMean[kChannels] = {0.485F, 0.456F, 0.406F};
constexpr float kStd[kChannels] = {0.229F, 0.224F, 0.225F};

#ifndef TTNN_CODEGEN_BATCH_SIZE
#define TTNN_CODEGEN_BATCH_SIZE 1
#endif
#ifndef TTNN_CODEGEN_GRAPH_DIR
#define TTNN_CODEGEN_GRAPH_DIR "."
#endif

struct Args {
  std::filesystem::path image;
  std::filesystem::path image_dir =
      "../../models/demos/vision/classification/resnet50/ttnn_resnet/demo/"
      "images";
  std::filesystem::path output;
  int batch_size = TTNN_CODEGEN_BATCH_SIZE;
  int warmup_iterations = 10;
  int measurement_iterations = 50;
  bool print_output = false;
};

struct DecodedImage {
  int width = 0;
  int height = 0;
  std::vector<uint8_t> rgb;
};

struct Timing {
  double preprocess_seconds = 0.0;
  double inference_seconds = 0.0;
  double total_seconds = 0.0;
};

struct CsvRow {
  std::string batch;
  std::string workers;
  std::string total_run_time;
  std::string preprocessing_time;
  std::string inference_time;
  std::string cpu_inference_time;
  std::string tt_inference_throughput;
  std::string total_throughput_with_preprocessing;
  std::string cpu_throughput;
  std::string total_images_processed;
  std::string tt_inference_iterations;
  std::string cpu_iterations;
  std::string status;
};

void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) ||          \
    defined(_M_IX86)
  _mm_pause();
#elif defined(__aarch64__) || defined(__arm__)
  asm volatile("yield" ::: "memory");
#else
  std::this_thread::yield();
#endif
}

template <typename Predicate> void spin_until(Predicate predicate) {
  while (!predicate()) {
    cpu_relax();
  }
}

class TurboJpegDecoder {
public:
  TurboJpegDecoder() : handle_(tjInitDecompress()) {
    if (handle_ == nullptr) {
      throw std::runtime_error(std::string("tjInitDecompress failed: ") +
                               tjGetErrorStr());
    }
  }

  TurboJpegDecoder(const TurboJpegDecoder &) = delete;
  TurboJpegDecoder &operator=(const TurboJpegDecoder &) = delete;

  ~TurboJpegDecoder() {
    if (handle_ != nullptr) {
      tjDestroy(handle_);
    }
  }

  tjhandle get() const { return handle_; }

private:
  tjhandle handle_ = nullptr;
};

int python_round(float value) { return static_cast<int>(std::nearbyint(value)); }

DecodedImage decode_rgb(tjhandle handle, const std::string &jpeg_bytes) {
  int width = 0;
  int height = 0;
  int subsampling = 0;
  int colorspace = 0;
  const auto *jpeg_data =
      reinterpret_cast<const unsigned char *>(jpeg_bytes.data());
  const auto jpeg_size = static_cast<unsigned long>(jpeg_bytes.size());

  if (tjDecompressHeader3(handle, jpeg_data, jpeg_size, &width, &height,
                          &subsampling, &colorspace) != 0) {
    throw std::runtime_error(std::string("tjDecompressHeader3 failed: ") +
                             tjGetErrorStr());
  }

  DecodedImage image;
  image.width = width;
  image.height = height;
  image.rgb.resize(static_cast<size_t>(width) * static_cast<size_t>(height) *
                   kChannels);

  if (tjDecompress2(handle, jpeg_data, jpeg_size, image.rgb.data(), width, 0,
                    height, TJPF_RGB, 0) != 0) {
    throw std::runtime_error(std::string("tjDecompress2 failed: ") +
                             tjGetErrorStr());
  }

  return image;
}

float bilinear_rgb_sample(const DecodedImage &image, int channel, float y,
                          float x) {
  y = std::max(0.0F, std::min(y, static_cast<float>(image.height - 1)));
  x = std::max(0.0F, std::min(x, static_cast<float>(image.width - 1)));

  const int y0 = static_cast<int>(std::floor(y));
  const int x0 = static_cast<int>(std::floor(x));
  const int y1 = std::min(y0 + 1, image.height - 1);
  const int x1 = std::min(x0 + 1, image.width - 1);
  const float wy = y - static_cast<float>(y0);
  const float wx = x - static_cast<float>(x0);

  auto pixel = [&](int py, int px) -> float {
    const size_t offset =
        (static_cast<size_t>(py) * image.width + px) * kChannels + channel;
    return static_cast<float>(image.rgb[offset]);
  };

  const float top = pixel(y0, x0) * (1.0F - wx) + pixel(y0, x1) * wx;
  const float bottom = pixel(y1, x0) * (1.0F - wx) + pixel(y1, x1) * wx;
  return top * (1.0F - wy) + bottom * wy;
}

void preprocess_one_into_float_tensor(tjhandle jpeg_handle,
                                      const std::string &jpeg_bytes,
                                      float *output, int batch_index) {
  const DecodedImage image = decode_rgb(jpeg_handle, jpeg_bytes);

  int resized_height = 0;
  int resized_width = 0;
  if (image.height <= image.width) {
    resized_height = kResizeShorterSide;
    resized_width = python_round(static_cast<float>(image.width) *
                                 kResizeShorterSide / image.height);
  } else {
    resized_height = python_round(static_cast<float>(image.height) *
                                  kResizeShorterSide / image.width);
    resized_width = kResizeShorterSide;
  }

  const int top = (resized_height - kCropSize) / 2;
  const int left = (resized_width - kCropSize) / 2;
  const float scale_y = static_cast<float>(image.height) / resized_height;
  const float scale_x = static_cast<float>(image.width) / resized_width;
  const size_t image_stride =
      static_cast<size_t>(kChannels) * kCropSize * kCropSize;
  float *image_output = output + static_cast<size_t>(batch_index) * image_stride;

  for (int channel = 0; channel < kChannels; ++channel) {
    float *channel_output =
        image_output + static_cast<size_t>(channel) * kCropSize * kCropSize;
    for (int crop_y = 0; crop_y < kCropSize; ++crop_y) {
      const int resized_y = top + crop_y;
      const float source_y =
          (static_cast<float>(resized_y) + 0.5F) * scale_y - 0.5F;
      for (int crop_x = 0; crop_x < kCropSize; ++crop_x) {
        const int resized_x = left + crop_x;
        const float source_x =
            (static_cast<float>(resized_x) + 0.5F) * scale_x - 0.5F;
        const float pixel =
            bilinear_rgb_sample(image, channel, source_y, source_x);
        channel_output[static_cast<size_t>(crop_y) * kCropSize + crop_x] =
            (pixel * (1.0F / 255.0F) - kMean[channel]) / kStd[channel];
      }
    }
  }
}

std::vector<float> preprocess_batch(const std::vector<std::string> &image_bytes,
                                    int batch_size, int workers,
                                    int iteration) {
  if (image_bytes.empty()) {
    throw std::invalid_argument("image list must not be empty");
  }

  const int actual_workers = std::min(workers, batch_size);
  std::vector<float> output(static_cast<size_t>(batch_size) * kChannels *
                            kCropSize * kCropSize);

  std::atomic<bool> start_work{false};
  std::atomic<bool> failed{false};
  std::atomic<int> initialized_workers{0};
  std::atomic<int> completed_workers{0};
  std::exception_ptr first_exception = nullptr;
  std::mutex exception_mutex;

  auto set_exception = [&]() {
    std::lock_guard<std::mutex> lock(exception_mutex);
    if (first_exception == nullptr) {
      first_exception = std::current_exception();
      failed.store(true, std::memory_order_release);
    }
  };

  auto worker_fn = [&](int worker_index) {
    try {
      TurboJpegDecoder decoder;
      initialized_workers.fetch_add(1, std::memory_order_release);
      spin_until([&]() { return start_work.load(std::memory_order_acquire); });

      if (!failed.load(std::memory_order_acquire)) {
        const int begin = worker_index * batch_size / actual_workers;
        const int end = (worker_index + 1) * batch_size / actual_workers;
        for (int batch_index = begin; batch_index < end; ++batch_index) {
          const int image_index =
              (iteration * batch_size + batch_index) %
              static_cast<int>(image_bytes.size());
          preprocess_one_into_float_tensor(decoder.get(), image_bytes[image_index],
                                           output.data(), batch_index);
        }
      }
    } catch (...) {
      set_exception();
    }
    completed_workers.fetch_add(1, std::memory_order_release);
  };

  std::vector<std::thread> threads;
  threads.reserve(static_cast<size_t>(actual_workers));
  for (int worker = 0; worker < actual_workers; ++worker) {
    threads.emplace_back(worker_fn, worker);
  }

  spin_until([&]() {
    return initialized_workers.load(std::memory_order_acquire) ==
               actual_workers ||
           failed.load(std::memory_order_acquire);
  });
  start_work.store(true, std::memory_order_release);
  spin_until([&]() {
    return completed_workers.load(std::memory_order_acquire) == actual_workers;
  });

  for (auto &thread : threads) {
    if (thread.joinable()) {
      thread.join();
    }
  }

  {
    std::lock_guard<std::mutex> lock(exception_mutex);
    if (first_exception != nullptr) {
      std::rethrow_exception(first_exception);
    }
  }

  return output;
}

::ttnn::Tensor make_device_input_tensor(
    std::vector<float> &&input, ttnn::distributed::MeshDevice *device,
    int batch_size) {
  ::ttnn::TensorSpec spec(
      ::ttnn::Shape{static_cast<uint32_t>(batch_size), kChannels, kCropSize,
                    kCropSize},
      ::tt::tt_metal::TensorLayout(
          ::ttnn::DataType::FLOAT32,
          ::tt::tt_metal::PageConfig(::ttnn::Layout::ROW_MAJOR),
          ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                               ::ttnn::BufferType::SYSTEM_MEMORY,
                               ::std::nullopt}));
  ::ttnn::Tensor host_tensor = ::ttnn::Tensor::from_vector(std::move(input), spec);
  return ::ttnn::to_device(
      host_tensor, device,
      ::ttnn::MemoryConfig{::ttnn::TensorMemoryLayout::INTERLEAVED,
                           ::ttnn::BufferType::DRAM, ::std::nullopt});
}

std::string read_file_bytes(const std::filesystem::path &path) {
  std::ifstream file(path, std::ios::binary);
  if (!file) {
    throw std::runtime_error("failed to open " + path.string());
  }
  return std::string(std::istreambuf_iterator<char>(file),
                     std::istreambuf_iterator<char>());
}

std::vector<std::string> load_image_bytes(const Args &args) {
  if (!args.image.empty()) {
    return {read_file_bytes(args.image)};
  }

  std::vector<std::filesystem::path> image_paths;
  for (const auto &entry : std::filesystem::directory_iterator(args.image_dir)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    std::string ext = entry.path().extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (ext == ".jpg" || ext == ".jpeg") {
      image_paths.push_back(entry.path());
    }
  }
  std::sort(image_paths.begin(), image_paths.end());
  if (image_paths.empty()) {
    throw std::runtime_error("no jpg/jpeg images found in " +
                             args.image_dir.string());
  }

  std::vector<std::string> images;
  images.reserve(image_paths.size());
  for (const auto &path : image_paths) {
    images.push_back(read_file_bytes(path));
  }
  return images;
}

std::string format_decimal(double value, int digits = 4) {
  std::ostringstream stream;
  stream << std::fixed << std::setprecision(digits) << value;
  return stream.str();
}

std::string format_throughput(int batch_size, double seconds) {
  if (seconds == 0.0) {
    return "";
  }
  return format_decimal(static_cast<double>(batch_size) / seconds);
}

std::string csv_escape(const std::string &value) {
  if (value.find_first_of(",\"\n") == std::string::npos) {
    return value;
  }
  std::string escaped = "\"";
  for (char c : value) {
    if (c == '"') {
      escaped += "\"\"";
    } else {
      escaped += c;
    }
  }
  escaped += "\"";
  return escaped;
}

void write_csv_row(std::ofstream &file, const std::vector<std::string> &values) {
  for (size_t index = 0; index < values.size(); ++index) {
    if (index != 0) {
      file << ',';
    }
    file << csv_escape(values[index]);
  }
  file << '\n';
}

std::filesystem::path default_output_path(const char *argv0) {
  const auto now = std::chrono::system_clock::now();
  const std::time_t now_time = std::chrono::system_clock::to_time_t(now);
  const auto nanos = std::chrono::duration_cast<std::chrono::nanoseconds>(
                         now.time_since_epoch())
                         .count() %
                     1000000000LL;

  std::tm local_time{};
#if defined(_WIN32)
  localtime_s(&local_time, &now_time);
#else
  localtime_r(&now_time, &local_time);
#endif

  std::ostringstream timestamp;
  timestamp << std::put_time(&local_time, "%Y%m%d_%H%M%S") << '_'
            << std::setw(9) << std::setfill('0') << nanos;

  const std::filesystem::path executable_dir =
      std::filesystem::absolute(std::filesystem::path(argv0)).parent_path();
  return executable_dir / "reports" /
         ("resnet50_blackhole_trace_1cq_cpp_preprocessing_e2e_" +
          timestamp.str() + ".csv");
}

void write_report(const std::filesystem::path &output_path,
                  const std::vector<CsvRow> &rows) {
  if (!output_path.parent_path().empty()) {
    std::filesystem::create_directories(output_path.parent_path());
  }
  std::ofstream file(output_path);
  if (!file) {
    throw std::runtime_error("failed to write CSV report to " +
                             output_path.string());
  }

  write_csv_row(
      file,
      {"Batch",
       "Workers",
       "Total Run Time (sec)",
       "Pre-processing Time (sec)",
       "Inference Time (sec)",
       "CPU Inference Time (sec)",
       "TT Inference Throughput (images/sec)",
       "Total Throughput With Pre-processing (images/sec)",
       "CPU Throughput (images/sec)",
       "Total Images Processed",
       "TT Inference Iterations",
       "CPU Iterations",
       "Status"});
  const CsvRow *best_row = nullptr;
  for (const CsvRow &row : rows) {
    write_csv_row(file,
                  {row.batch,
                   row.workers,
                   row.total_run_time,
                   row.preprocessing_time,
                   row.inference_time,
                   row.cpu_inference_time,
                   row.tt_inference_throughput,
                   row.total_throughput_with_preprocessing,
                   row.cpu_throughput,
                   row.total_images_processed,
                   row.tt_inference_iterations,
                   row.cpu_iterations,
                   row.status});
    if (row.status == "OK" &&
        (best_row == nullptr ||
         std::stod(row.total_throughput_with_preprocessing) >
             std::stod(best_row->total_throughput_with_preprocessing))) {
      best_row = &row;
    }
  }
  write_csv_row(file, {});
  if (best_row != nullptr) {
    write_csv_row(file, {"Best Workers By Total Throughput With Pre-processing",
                         best_row->workers});
    write_csv_row(file,
                  {"Best Aggregate Total Throughput With Pre-processing "
                   "(images/sec)",
                   best_row->total_throughput_with_preprocessing});
  } else {
    write_csv_row(file,
                  {"Best Workers By Total Throughput With Pre-processing",
                   "unknown"});
  }
}

int parse_int_arg(const char *value, const std::string &name) {
  try {
    return std::stoi(value);
  } catch (const std::exception &) {
    throw std::invalid_argument("invalid integer for " + name + ": " + value);
  }
}

Args parse_args(int argc, char **argv) {
  Args args;
  args.workers = static_cast<int>(std::max(1U, std::thread::hardware_concurrency()));

  for (int index = 1; index < argc; ++index) {
    const std::string flag = argv[index];
    auto require_value = [&](const std::string &name) -> const char * {
      if (index + 1 >= argc) {
        throw std::invalid_argument(name + " requires a value");
      }
      return argv[++index];
    };

    if (flag == "--image") {
      args.image = require_value(flag);
    } else if (flag == "--image-dir") {
      args.image_dir = require_value(flag);
    } else if (flag == "--output") {
      args.output = require_value(flag);
    } else if (flag == "--warmup-iterations") {
      args.warmup_iterations = parse_int_arg(require_value(flag), flag);
    } else if (flag == "--measurement-iterations") {
      args.measurement_iterations = parse_int_arg(require_value(flag), flag);
    } else if (flag == "--print-output") {
      args.print_output = true;
    } else if (flag == "--help" || flag == "-h") {
      std::cout
          << "Usage: " << argv[0]
          << " [--image path | --image-dir dir] [--warmup-iterations n]\n"
          << "       [--measurement-iterations n] [--output path] "
             "[--print-output]\n"
          << "\nThis binary is compiled for --batch-size "
          << TTNN_CODEGEN_BATCH_SIZE << ".\n";
      std::exit(0);
    } else {
      throw std::invalid_argument("unknown argument: " + flag);
    }
  }

  if (args.batch_size != TTNN_CODEGEN_BATCH_SIZE) {
    throw std::invalid_argument(
        "requested batch size does not match the compiled codegen graph batch "
        "size");
  }
  if (args.warmup_iterations < 0 || args.measurement_iterations <= 0) {
    throw std::invalid_argument(
        "warmup must be >= 0 and measurement iterations must be > 0");
  }
  return args;
}

void maybe_print_top5(const ::ttnn::Tensor &device_output) {
  ::ttnn::Tensor host = ::ttnn::from_device(device_output);
  if (host.layout() != ::ttnn::Layout::ROW_MAJOR) {
    host = ::ttnn::to_layout(host, ::ttnn::Layout::ROW_MAJOR);
  }
  std::vector<float> logits = host.to_vector<float>();
  std::vector<int> indexes(logits.size());
  for (size_t i = 0; i < indexes.size(); ++i) {
    indexes[i] = static_cast<int>(i);
  }
  const size_t top_k = std::min<size_t>(5, indexes.size());
  std::partial_sort(indexes.begin(), indexes.begin() + top_k, indexes.end(),
                    [&](int a, int b) { return logits[a] > logits[b]; });
  for (size_t rank = 0; rank < top_k; ++rank) {
    const int class_index = indexes[rank];
    std::cout << "top" << (rank + 1) << " class_index=" << class_index
              << " logit=" << logits[class_index] << "\n";
  }
}

Timing run_one_iteration(std::vector<::ttnn::Tensor> &inputs,
                         const std::vector<std::string> &image_bytes,
                         const Args &args, int workers, int iteration,
                         std::optional<::ttnn::Tensor> &last_output) {
  ttnn::distributed::MeshDevice *device = ttnn::DeviceGetter::getInstance();

  const auto total_start = std::chrono::steady_clock::now();
  const auto preprocess_start = total_start;
  std::vector<float> host_input =
      preprocess_batch(image_bytes, args.batch_size, workers, iteration);
  const auto preprocess_end = std::chrono::steady_clock::now();

  const auto inference_start = preprocess_end;
  inputs[kInputTensorIndex] = make_device_input_tensor(std::move(host_input), device, args.batch_size);
  ::std::vector<::ttnn::Tensor> outputs = _main(inputs);
  tt::tt_metal::distributed::Synchronize(device, std::nullopt);
  const auto inference_end = std::chrono::steady_clock::now();

  if (last_output.has_value()) {
    ::ttnn::deallocate(*last_output, false);
  }
  last_output = outputs[0];

  return Timing{
      std::chrono::duration<double>(preprocess_end - preprocess_start).count(),
      std::chrono::duration<double>(inference_end - inference_start).count(),
      std::chrono::duration<double>(inference_end - total_start).count()};
}

} // namespace

int main(int argc, char **argv) {
  try {
    const Args args = parse_args(argc, argv);
    const std::filesystem::path output_path =
        args.output.empty() ? default_output_path(argv[0])
                            : std::filesystem::absolute(args.output);
    std::filesystem::current_path(TTNN_CODEGEN_GRAPH_DIR);
    const std::vector<std::string> image_bytes = load_image_bytes(args);
    std::vector<::ttnn::Tensor> inputs =
        load_inputs_for__main("./tensors/arg27.tensorbin");
    std::optional<::ttnn::Tensor> last_output;
    const std::vector<int> worker_counts = {1, 2, 4, 8, 16, 32};
    std::vector<CsvRow> rows;
    rows.reserve(worker_counts.size());

    for (int workers : worker_counts) {
      for (int iteration = 0; iteration < args.warmup_iterations; ++iteration) {
        run_one_iteration(inputs, image_bytes, args, workers, iteration,
                          last_output);
      }

      double preprocess_total = 0.0;
      double inference_total = 0.0;
      double total_total = 0.0;
      for (int iteration = 0; iteration < args.measurement_iterations;
           ++iteration) {
        const Timing timing =
            run_one_iteration(inputs, image_bytes, args, workers,
                              args.warmup_iterations + iteration, last_output);
        preprocess_total += timing.preprocess_seconds;
        inference_total += timing.inference_seconds;
        total_total += timing.total_seconds;
      }

      const double iterations =
          static_cast<double>(args.measurement_iterations);
      const double preprocess_mean = preprocess_total / iterations;
      const double inference_mean = inference_total / iterations;
      const double total_mean = total_total / iterations;
      rows.push_back(CsvRow{
          std::to_string(args.batch_size),
          std::to_string(workers),
          format_decimal(total_mean),
          format_decimal(preprocess_mean),
          format_decimal(inference_mean),
          "",
          format_throughput(args.batch_size, inference_mean),
          format_throughput(args.batch_size, total_mean),
          "",
          std::to_string(args.batch_size * args.measurement_iterations),
          std::to_string(args.measurement_iterations),
          "",
          "OK"});

      std::cout << "batch_size," << args.batch_size << "\n";
      std::cout << "workers," << workers << "\n";
      std::cout << "warmup_iterations," << args.warmup_iterations << "\n";
      std::cout << "measurement_iterations," << args.measurement_iterations
                << "\n";
      std::cout << "preprocess_mean_sec," << preprocess_mean << "\n";
      std::cout << "inference_mean_sec," << inference_mean << "\n";
      std::cout << "e2e_mean_sec," << total_mean << "\n";
      std::cout << "e2e_images_per_sec,"
                << static_cast<double>(args.batch_size) / total_mean << "\n";
    }

    write_report(output_path, rows);
    std::cout << "csv_report," << output_path << "\n";

    if (args.print_output && last_output.has_value()) {
      maybe_print_top5(*last_output);
    }
    if (last_output.has_value()) {
      ::ttnn::deallocate(*last_output, false);
    }
  } catch (const std::exception &error) {
    std::cerr << "error: " << error.what() << "\n";
    return 1;
  }
  return 0;
}
