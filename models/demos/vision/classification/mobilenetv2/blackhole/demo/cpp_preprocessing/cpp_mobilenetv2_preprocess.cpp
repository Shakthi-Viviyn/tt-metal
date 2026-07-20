// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <torch/extension.h>

#include <turbojpeg.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <exception>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
#include <immintrin.h>
#endif

namespace py = pybind11;

namespace {

constexpr int64_t kBatchSize = 10;
constexpr int kResizeShorterSide = 224;
constexpr int kImageSize = 224;
constexpr int kInputChannels = 3;
constexpr int kPaddedChannels = 16;
constexpr int kPackedChannels = 32;
constexpr int64_t kPixelsPerImage = static_cast<int64_t>(kImageSize) * kImageSize;
constexpr float kMean[kInputChannels] = {0.485F, 0.456F, 0.406F};
constexpr float kStd[kInputChannels] = {0.229F, 0.224F, 0.225F};

void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
    _mm_pause();
#elif defined(__aarch64__) || defined(__arm__)
    asm volatile("yield" ::: "memory");
#else
    std::this_thread::yield();
#endif
}

template <typename Predicate>
void spin_until(Predicate predicate) {
    while (!predicate()) {
        cpu_relax();
    }
}

struct DecodedImage {
    int width = 0;
    int height = 0;
    std::vector<uint8_t> rgb;
};

int python_round(float value) { return static_cast<int>(std::nearbyint(value)); }

class TurboJpegDecoder {
public:
    TurboJpegDecoder() : handle_(tjInitDecompress()) {
        if (handle_ == nullptr) {
            throw std::runtime_error(std::string("tjInitDecompress failed: ") + tjGetErrorStr());
        }
    }

    TurboJpegDecoder(const TurboJpegDecoder&) = delete;
    TurboJpegDecoder& operator=(const TurboJpegDecoder&) = delete;

    ~TurboJpegDecoder() {
        if (handle_ != nullptr) {
            tjDestroy(handle_);
        }
    }

    tjhandle get() const { return handle_; }

private:
    tjhandle handle_ = nullptr;
};

DecodedImage decode_rgb(tjhandle handle, const std::string& jpeg_bytes) {
    int width = 0;
    int height = 0;
    int subsampling = 0;
    int colorspace = 0;
    auto* jpeg_data = reinterpret_cast<const unsigned char*>(jpeg_bytes.data());
    const auto jpeg_size = static_cast<unsigned long>(jpeg_bytes.size());

    if (tjDecompressHeader3(handle, jpeg_data, jpeg_size, &width, &height, &subsampling, &colorspace) != 0) {
        throw std::runtime_error(std::string("tjDecompressHeader3 failed: ") + tjGetErrorStr());
    }

    DecodedImage image;
    image.width = width;
    image.height = height;
    image.rgb.resize(static_cast<size_t>(width) * static_cast<size_t>(height) * kInputChannels);

    if (tjDecompress2(handle, jpeg_data, jpeg_size, image.rgb.data(), width, 0, height, TJPF_RGB, 0) != 0) {
        throw std::runtime_error(std::string("tjDecompress2 failed: ") + tjGetErrorStr());
    }

    return image;
}

float bilinear_rgb_sample(const DecodedImage& image, int channel, float y, float x) {
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
            (static_cast<size_t>(py) * image.width + px) * kInputChannels + static_cast<size_t>(channel);
        return static_cast<float>(image.rgb[offset]);
    };

    const float top = pixel(y0, x0) * (1.0F - wx) + pixel(y0, x1) * wx;
    const float bottom = pixel(y1, x0) * (1.0F - wx) + pixel(y1, x1) * wx;
    return top * (1.0F - wy) + bottom * wy;
}

void preprocess_one_into_tensors(
    tjhandle jpeg_handle,
    const std::string& jpeg_bytes,
    float* nchw_output,
    at::BFloat16* packed_output,
    int64_t batch_index) {
    const DecodedImage image = decode_rgb(jpeg_handle, jpeg_bytes);

    int resized_height = 0;
    int resized_width = 0;
    if (image.height <= image.width) {
        resized_height = kResizeShorterSide;
        resized_width = python_round(static_cast<float>(image.width) * kResizeShorterSide / image.height);
    } else {
        resized_height = python_round(static_cast<float>(image.height) * kResizeShorterSide / image.width);
        resized_width = kResizeShorterSide;
    }

    const int top = (resized_height - kImageSize) / 2;
    const int left = (resized_width - kImageSize) / 2;
    const float scale_y = static_cast<float>(image.height) / resized_height;
    const float scale_x = static_cast<float>(image.width) / resized_width;
    float* image_nchw_output = nchw_output + batch_index * kInputChannels * kPixelsPerImage;

    for (int output_y = 0; output_y < kImageSize; ++output_y) {
        const int resized_y = top + output_y;
        const float source_y = (static_cast<float>(resized_y) + 0.5F) * scale_y - 0.5F;
        for (int output_x = 0; output_x < kImageSize; ++output_x) {
            const int resized_x = left + output_x;
            const float source_x = (static_cast<float>(resized_x) + 0.5F) * scale_x - 0.5F;
            const int64_t image_pixel = static_cast<int64_t>(output_y) * kImageSize + output_x;
            const int64_t flat_pixel = batch_index * kPixelsPerImage + image_pixel;
            at::BFloat16* packed_pixel =
                packed_output + (flat_pixel / 2) * kPackedChannels + (flat_pixel % 2) * kPaddedChannels;

            for (int channel = 0; channel < kInputChannels; ++channel) {
                const float pixel = bilinear_rgb_sample(image, channel, source_y, source_x);
                const float normalized = (pixel * (1.0F / 255.0F) - kMean[channel]) / kStd[channel];
                image_nchw_output[static_cast<int64_t>(channel) * kPixelsPerImage + image_pixel] = normalized;
                packed_pixel[channel] = at::BFloat16(normalized);
            }
        }
    }
}

struct PreprocessResult {
    double seconds = 0.0;
    torch::Tensor nchw_inputs;
    torch::Tensor packed_inputs;
};

PreprocessResult measure_preprocessing_cpp(
    const std::vector<std::string>& image_bytes_list,
    int64_t batch_size,
    int64_t warmup_iterations,
    int64_t measurement_iterations,
    int64_t workers) {
    if (image_bytes_list.empty()) {
        throw std::invalid_argument("image_bytes_list must not be empty");
    }
    if (batch_size != kBatchSize) {
        throw std::invalid_argument("MobileNetV2 Blackhole preprocessing supports only batch_size=10");
    }
    if (workers <= 0) {
        throw std::invalid_argument("workers must be positive");
    }
    if (warmup_iterations < 0) {
        throw std::invalid_argument("warmup_iterations must be non-negative");
    }
    if (measurement_iterations <= 0) {
        throw std::invalid_argument("measurement_iterations must be positive");
    }

    const int64_t actual_workers = std::min<int64_t>(workers, batch_size);
    const int64_t total_iterations = warmup_iterations + measurement_iterations;
    torch::Tensor last_nchw_inputs;
    torch::Tensor last_packed_inputs;
    double total_seconds = 0.0;

    for (int64_t iteration = 0; iteration < total_iterations; ++iteration) {
        last_nchw_inputs = torch::empty(
            {batch_size, kInputChannels, kImageSize, kImageSize},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));
        last_packed_inputs = torch::zeros(
            {1, 1, batch_size * kPixelsPerImage / 2, kPackedChannels},
            torch::TensorOptions().dtype(torch::kBFloat16).device(torch::kCPU));

        auto* nchw_output_ptr = last_nchw_inputs.data_ptr<float>();
        auto* packed_output_ptr = last_packed_inputs.data_ptr<at::BFloat16>();

        std::atomic<bool> start_work{false};
        std::atomic<bool> failed{false};
        std::atomic<int64_t> initialized_workers{0};
        std::atomic<int64_t> completed_workers{0};
        std::exception_ptr first_exception = nullptr;
        std::mutex exception_mutex;

        auto set_exception = [&]() {
            std::lock_guard<std::mutex> lock(exception_mutex);
            if (first_exception == nullptr) {
                first_exception = std::current_exception();
                failed.store(true, std::memory_order_release);
            }
        };

        auto worker_fn = [&](int64_t worker_index) {
            try {
                TurboJpegDecoder decoder;
                initialized_workers.fetch_add(1, std::memory_order_release);
                spin_until([&]() { return start_work.load(std::memory_order_acquire); });

                if (!failed.load(std::memory_order_acquire)) {
                    const int64_t begin = worker_index * batch_size / actual_workers;
                    const int64_t end = (worker_index + 1) * batch_size / actual_workers;
                    for (int64_t batch_index = begin; batch_index < end; ++batch_index) {
                        const int64_t image_index =
                            (iteration * batch_size + batch_index) % static_cast<int64_t>(image_bytes_list.size());
                        preprocess_one_into_tensors(
                            decoder.get(),
                            image_bytes_list[image_index],
                            nchw_output_ptr,
                            packed_output_ptr,
                            batch_index);
                    }
                }
            } catch (...) {
                set_exception();
            }
            completed_workers.fetch_add(1, std::memory_order_release);
        };

        std::vector<std::thread> threads;
        threads.reserve(static_cast<size_t>(actual_workers));
        for (int64_t worker = 0; worker < actual_workers; ++worker) {
            threads.emplace_back(worker_fn, worker);
        }

        spin_until([&]() {
            return initialized_workers.load(std::memory_order_acquire) == actual_workers ||
                   failed.load(std::memory_order_acquire);
        });

        const auto start = std::chrono::steady_clock::now();
        start_work.store(true, std::memory_order_release);
        spin_until([&]() { return completed_workers.load(std::memory_order_acquire) == actual_workers; });
        const auto end = std::chrono::steady_clock::now();

        for (auto& thread : threads) {
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

        if (iteration >= warmup_iterations) {
            total_seconds += std::chrono::duration<double>(end - start).count();
        }
    }

    return {total_seconds / static_cast<double>(measurement_iterations), last_nchw_inputs, last_packed_inputs};
}

std::vector<std::string> py_bytes_list_to_vector(const py::list& py_image_bytes_list) {
    std::vector<std::string> image_bytes_list;
    image_bytes_list.reserve(py_image_bytes_list.size());
    for (const py::handle item : py_image_bytes_list) {
        image_bytes_list.push_back(py::cast<std::string>(item));
    }
    return image_bytes_list;
}

py::tuple measure_preprocessing(
    const py::list& py_image_bytes_list,
    int64_t batch_size,
    int64_t warmup_iterations,
    int64_t measurement_iterations,
    int64_t workers) {
    std::vector<std::string> image_bytes_list = py_bytes_list_to_vector(py_image_bytes_list);
    PreprocessResult result;
    {
        py::gil_scoped_release release;
        result =
            measure_preprocessing_cpp(image_bytes_list, batch_size, warmup_iterations, measurement_iterations, workers);
    }
    return py::make_tuple(result.seconds, result.nchw_inputs, result.packed_inputs);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.doc() = "C++ TurboJPEG preprocessing for the MobileNetV2 Blackhole benchmark";
    module.def(
        "measure_preprocessing",
        &measure_preprocessing,
        py::arg("image_bytes_list"),
        py::arg("batch_size"),
        py::arg("warmup_iterations"),
        py::arg("measurement_iterations"),
        py::arg("workers"),
        "Run warmup and measured batch-10 preprocessing in C++ and return "
        "(mean_seconds, nchw_inputs_float32, packed_inputs_bfloat16).");
}
