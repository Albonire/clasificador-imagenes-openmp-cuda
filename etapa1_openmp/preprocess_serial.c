/*
 * Serial image preprocessing pipeline for Etapa 1.
 *
 * Compile:
 *   gcc -Wall -Wextra -O2 -o preprocess_serial preprocess_serial.c -lm
 *
 * Usage:
 *   ./preprocess_serial
 *   ./preprocess_serial <raw_dataset_dir> <output_csv>
 *
 * Labels:
 *   dataset/raw/clase_0 -> 0 (ojos abiertos)
 *   dataset/raw/clase_1 -> 1 (ojos cerrados)
 */

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <ctype.h>
#include <dirent.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#define OUTPUT_WIDTH 64
#define OUTPUT_HEIGHT 64
#define OUTPUT_PIXELS (OUTPUT_WIDTH * OUTPUT_HEIGHT)
#define GRAY_R 0.299f
#define GRAY_G 0.587f
#define GRAY_B 0.114f
#define MAX_PIXEL_VALUE 255.0f

typedef struct {
    int width;
    int height;
    float *data;
} GrayImage;

typedef struct {
    char **items;
    size_t count;
    size_t capacity;
} PathList;

static int path_exists(const char *path)
{
    struct stat info;
    return stat(path, &info) == 0;
}

static int directory_exists(const char *path)
{
    struct stat info;
    return stat(path, &info) == 0 && (info.st_mode & S_IFDIR) != 0;
}

static char *duplicate_string(const char *text)
{
    size_t length = strlen(text) + 1;
    char *copy = (char *)malloc(length);
    if (copy == NULL) {
        return NULL;
    }

    memcpy(copy, text, length);
    return copy;
}

static char *join_path(const char *left, const char *right)
{
    size_t left_length = strlen(left);
    size_t right_length = strlen(right);
    int needs_separator = left_length > 0 && left[left_length - 1] != '/' &&
                          left[left_length - 1] != '\\';
    size_t total_length = left_length + (size_t)needs_separator + right_length + 1;
    char *path = (char *)malloc(total_length);

    if (path == NULL) {
        return NULL;
    }

    snprintf(path, total_length, "%s%s%s", left, needs_separator ? "/" : "", right);
    return path;
}

static void free_path_list(PathList *list)
{
    size_t i;

    for (i = 0; i < list->count; i++) {
        free(list->items[i]);
    }

    free(list->items);
    list->items = NULL;
    list->count = 0;
    list->capacity = 0;
}

static int append_path(PathList *list, const char *path)
{
    char **new_items;
    char *copy;
    size_t new_capacity;

    if (list->count == list->capacity) {
        new_capacity = list->capacity == 0 ? 32 : list->capacity * 2;
        new_items = (char **)realloc(list->items, new_capacity * sizeof(char *));
        if (new_items == NULL) {
            return 0;
        }

        list->items = new_items;
        list->capacity = new_capacity;
    }

    copy = duplicate_string(path);
    if (copy == NULL) {
        return 0;
    }

    list->items[list->count++] = copy;
    return 1;
}

static int compare_paths(const void *left, const void *right)
{
    const char *const *left_path = (const char *const *)left;
    const char *const *right_path = (const char *const *)right;
    return strcmp(*left_path, *right_path);
}

static int has_supported_extension(const char *name)
{
    const char *extension = strrchr(name, '.');
    char lower_extension[8];
    size_t i;

    if (extension == NULL || strlen(extension) >= sizeof(lower_extension)) {
        return 0;
    }

    for (i = 0; extension[i] != '\0'; i++) {
        lower_extension[i] = (char)tolower((unsigned char)extension[i]);
    }
    lower_extension[i] = '\0';

    return strcmp(lower_extension, ".jpg") == 0 ||
           strcmp(lower_extension, ".jpeg") == 0 ||
           strcmp(lower_extension, ".png") == 0 ||
           strcmp(lower_extension, ".bmp") == 0 ||
           strcmp(lower_extension, ".tga") == 0;
}

static int collect_image_paths(const char *class_dir, PathList *paths)
{
    DIR *dir = opendir(class_dir);
    struct dirent *entry;

    if (dir == NULL) {
        fprintf(stderr, "ERROR: could not open directory '%s': %s\n", class_dir,
                strerror(errno));
        return 0;
    }

    while ((entry = readdir(dir)) != NULL) {
        char *file_path;

        if (entry->d_name[0] == '.' || !has_supported_extension(entry->d_name)) {
            continue;
        }

        file_path = join_path(class_dir, entry->d_name);
        if (file_path == NULL) {
            closedir(dir);
            return 0;
        }

        if (path_exists(file_path) && !append_path(paths, file_path)) {
            free(file_path);
            closedir(dir);
            return 0;
        }

        free(file_path);
    }

    closedir(dir);
    qsort(paths->items, paths->count, sizeof(char *), compare_paths);
    return 1;
}

static float clamp_float(float value, float min_value, float max_value)
{
    if (value < min_value) {
        return min_value;
    }
    if (value > max_value) {
        return max_value;
    }
    return value;
}

static int clamp_int(int value, int min_value, int max_value)
{
    if (value < min_value) {
        return min_value;
    }
    if (value > max_value) {
        return max_value;
    }
    return value;
}

static GrayImage create_gray_image(int width, int height)
{
    GrayImage image;
    size_t total_pixels = (size_t)width * (size_t)height;

    image.width = width;
    image.height = height;
    image.data = (float *)malloc(total_pixels * sizeof(float));

    if (image.data == NULL) {
        image.width = 0;
        image.height = 0;
    }

    return image;
}

static void free_gray_image(GrayImage *image)
{
    free(image->data);
    image->data = NULL;
    image->width = 0;
    image->height = 0;
}

static GrayImage load_grayscale_image(const char *file_path)
{
    int width = 0;
    int height = 0;
    int channels = 0;
    unsigned char *pixels = stbi_load(file_path, &width, &height, &channels, 3);
    GrayImage gray;
    size_t total_pixels;
    size_t i;

    if (pixels == NULL) {
        fprintf(stderr, "ERROR: could not load image '%s' (%s)\n", file_path,
                stbi_failure_reason());
        gray.width = 0;
        gray.height = 0;
        gray.data = NULL;
        return gray;
    }

    gray = create_gray_image(width, height);
    if (gray.data == NULL) {
        fprintf(stderr, "ERROR: not enough memory for grayscale image '%s'\n", file_path);
        stbi_image_free(pixels);
        return gray;
    }

    total_pixels = (size_t)width * (size_t)height;
    for (i = 0; i < total_pixels; i++) {
        size_t offset = i * 3;
        float r = (float)pixels[offset];
        float g = (float)pixels[offset + 1];
        float b = (float)pixels[offset + 2];
        gray.data[i] = GRAY_R * r + GRAY_G * g + GRAY_B * b;
    }

    stbi_image_free(pixels);
    return gray;
}

static GrayImage apply_sobel_filter(const GrayImage *source)
{
    static const int kernel_x[3][3] = {
        {-1, 0, 1},
        {-2, 0, 2},
        {-1, 0, 1},
    };
    static const int kernel_y[3][3] = {
        {-1, -2, -1},
        {0, 0, 0},
        {1, 2, 1},
    };
    GrayImage result = create_gray_image(source->width, source->height);
    int y;

    if (result.data == NULL) {
        return result;
    }

    for (y = 0; y < source->height; y++) {
        int x;
        for (x = 0; x < source->width; x++) {
            float gradient_x = 0.0f;
            float gradient_y = 0.0f;
            int ky;

            for (ky = -1; ky <= 1; ky++) {
                int kx;
                int sample_y = clamp_int(y + ky, 0, source->height - 1);

                for (kx = -1; kx <= 1; kx++) {
                    int sample_x = clamp_int(x + kx, 0, source->width - 1);
                    float pixel = source->data[(size_t)sample_y * (size_t)source->width +
                                               (size_t)sample_x];
                    gradient_x += pixel * (float)kernel_x[ky + 1][kx + 1];
                    gradient_y += pixel * (float)kernel_y[ky + 1][kx + 1];
                }
            }

            result.data[(size_t)y * (size_t)source->width + (size_t)x] =
                clamp_float(sqrtf(gradient_x * gradient_x + gradient_y * gradient_y),
                            0.0f, MAX_PIXEL_VALUE);
        }
    }

    return result;
}

static GrayImage apply_gaussian_filter(const GrayImage *source)
{
    static const int kernel[3][3] = {
        {1, 2, 1},
        {2, 4, 2},
        {1, 2, 1},
    };
    GrayImage result = create_gray_image(source->width, source->height);
    int y;

    if (result.data == NULL) {
        return result;
    }

    for (y = 0; y < source->height; y++) {
        int x;
        for (x = 0; x < source->width; x++) {
            float sum = 0.0f;
            int ky;

            for (ky = -1; ky <= 1; ky++) {
                int kx;
                int sample_y = clamp_int(y + ky, 0, source->height - 1);

                for (kx = -1; kx <= 1; kx++) {
                    int sample_x = clamp_int(x + kx, 0, source->width - 1);
                    float pixel = source->data[(size_t)sample_y * (size_t)source->width +
                                               (size_t)sample_x];
                    sum += pixel * (float)kernel[ky + 1][kx + 1];
                }
            }

            result.data[(size_t)y * (size_t)source->width + (size_t)x] = sum / 16.0f;
        }
    }

    return result;
}

static GrayImage resize_bilinear(const GrayImage *source, int new_width, int new_height)
{
    GrayImage result = create_gray_image(new_width, new_height);
    float x_ratio = new_width > 1 ? (float)(source->width - 1) / (float)(new_width - 1) : 0.0f;
    float y_ratio = new_height > 1 ? (float)(source->height - 1) / (float)(new_height - 1) : 0.0f;
    int y;

    if (result.data == NULL) {
        return result;
    }

    for (y = 0; y < new_height; y++) {
        int x;
        float source_y = (float)y * y_ratio;
        int y0 = (int)floorf(source_y);
        int y1 = clamp_int(y0 + 1, 0, source->height - 1);
        float y_weight = source_y - (float)y0;

        for (x = 0; x < new_width; x++) {
            float source_x = (float)x * x_ratio;
            int x0 = (int)floorf(source_x);
            int x1 = clamp_int(x0 + 1, 0, source->width - 1);
            float x_weight = source_x - (float)x0;
            float top_left = source->data[(size_t)y0 * (size_t)source->width + (size_t)x0];
            float top_right = source->data[(size_t)y0 * (size_t)source->width + (size_t)x1];
            float bottom_left = source->data[(size_t)y1 * (size_t)source->width + (size_t)x0];
            float bottom_right = source->data[(size_t)y1 * (size_t)source->width + (size_t)x1];
            float top = top_left + (top_right - top_left) * x_weight;
            float bottom = bottom_left + (bottom_right - bottom_left) * x_weight;

            result.data[(size_t)y * (size_t)new_width + (size_t)x] =
                top + (bottom - top) * y_weight;
        }
    }

    return result;
}

static int preprocess_image(const char *file_path, float output[OUTPUT_PIXELS])
{
    GrayImage gray = load_grayscale_image(file_path);
    GrayImage sobel;
    GrayImage gaussian;
    GrayImage resized;
    size_t i;

    if (gray.data == NULL) {
        return 0;
    }

    sobel = apply_sobel_filter(&gray);
    free_gray_image(&gray);
    if (sobel.data == NULL) {
        fprintf(stderr, "ERROR: not enough memory while applying Sobel to '%s'\n",
                file_path);
        return 0;
    }

    gaussian = apply_gaussian_filter(&sobel);
    free_gray_image(&sobel);
    if (gaussian.data == NULL) {
        fprintf(stderr, "ERROR: not enough memory while applying Gaussian to '%s'\n",
                file_path);
        return 0;
    }

    resized = resize_bilinear(&gaussian, OUTPUT_WIDTH, OUTPUT_HEIGHT);
    free_gray_image(&gaussian);
    if (resized.data == NULL) {
        fprintf(stderr, "ERROR: not enough memory while resizing '%s'\n", file_path);
        return 0;
    }

    for (i = 0; i < OUTPUT_PIXELS; i++) {
        output[i] = clamp_float(resized.data[i], 0.0f, MAX_PIXEL_VALUE) / MAX_PIXEL_VALUE;
    }

    free_gray_image(&resized);
    return 1;
}

static int write_csv_header(FILE *csv)
{
    int i;

    if (fprintf(csv, "label") < 0) {
        return 0;
    }

    for (i = 0; i < OUTPUT_PIXELS; i++) {
        if (fprintf(csv, ",pixel_%d", i) < 0) {
            return 0;
        }
    }

    return fprintf(csv, "\n") >= 0;
}

static int write_csv_row(FILE *csv, int label, const float pixels[OUTPUT_PIXELS])
{
    int i;

    if (fprintf(csv, "%d", label) < 0) {
        return 0;
    }

    for (i = 0; i < OUTPUT_PIXELS; i++) {
        if (fprintf(csv, ",%.6f", pixels[i]) < 0) {
            return 0;
        }
    }

    return fprintf(csv, "\n") >= 0;
}

static int process_class_directory(FILE *csv, const char *raw_root, const char *class_name,
                                   int label, size_t *processed_count)
{
    char *class_dir = join_path(raw_root, class_name);
    PathList paths = {0};
    float pixels[OUTPUT_PIXELS];
    size_t i;

    if (class_dir == NULL) {
        return 0;
    }

    if (!directory_exists(class_dir)) {
        fprintf(stderr, "ERROR: class directory does not exist: %s\n", class_dir);
        free(class_dir);
        return 0;
    }

    if (!collect_image_paths(class_dir, &paths)) {
        free(class_dir);
        free_path_list(&paths);
        return 0;
    }

    printf("Class %s (label=%d): %zu images\n", class_name, label, paths.count);

    for (i = 0; i < paths.count; i++) {
        if (!preprocess_image(paths.items[i], pixels)) {
            fprintf(stderr, "WARNING: skipping image '%s'\n", paths.items[i]);
            continue;
        }

        if (!write_csv_row(csv, label, pixels)) {
            fprintf(stderr, "ERROR: could not write CSV row for '%s'\n", paths.items[i]);
            free(class_dir);
            free_path_list(&paths);
            return 0;
        }

        (*processed_count)++;
    }

    free(class_dir);
    free_path_list(&paths);
    return 1;
}

int main(int argc, char *argv[])
{
    const char *raw_root;
    const char *output_csv;
    FILE *csv;
    size_t processed_count = 0;

    if (argc != 1 && argc != 3) {
        fprintf(stderr, "Usage: %s [<raw_dataset_dir> <output_csv>]\n", argv[0]);
        fprintf(stderr, "Example: %s ../dataset/raw ../dataset/procesado/dataset_serial.csv\n",
                argv[0]);
        return EXIT_FAILURE;
    }

    if (argc == 3) {
        raw_root = argv[1];
        output_csv = argv[2];
    } else if (directory_exists("../dataset/raw")) {
        raw_root = "../dataset/raw";
        output_csv = "../dataset/procesado/dataset_serial.csv";
    } else {
        raw_root = "dataset/raw";
        output_csv = "dataset/procesado/dataset_serial.csv";
    }

    csv = fopen(output_csv, "w");
    if (csv == NULL) {
        fprintf(stderr, "ERROR: could not create output CSV '%s': %s\n", output_csv,
                strerror(errno));
        return EXIT_FAILURE;
    }

    if (!write_csv_header(csv)) {
        fprintf(stderr, "ERROR: could not write CSV header\n");
        fclose(csv);
        return EXIT_FAILURE;
    }

    if (!process_class_directory(csv, raw_root, "clase_0", 0, &processed_count) ||
        !process_class_directory(csv, raw_root, "clase_1", 1, &processed_count)) {
        fclose(csv);
        return EXIT_FAILURE;
    }

    if (fclose(csv) != 0) {
        fprintf(stderr, "ERROR: could not close output CSV '%s': %s\n", output_csv,
                strerror(errno));
        return EXIT_FAILURE;
    }

    printf("OK: processed %zu images into %s\n", processed_count, output_csv);
    return processed_count > 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
