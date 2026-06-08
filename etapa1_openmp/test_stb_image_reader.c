/*
 *
 * Objetivo: verificar que stb_image.h carga imagenes correctamente antes de
 * construir el pipeline de preprocesamiento en C/OpenMP.
 *
 * Compilar:
 *   gcc -Wall -Wextra -O2 -o test_stb_image_reader test_stb_image_reader.c -lm
 *
 * Uso:
 *   ./test_stb_image_reader <image_path>
 *   ./test_stb_image_reader ../dataset/raw/clase_0/PNG1.png
 */

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <stdio.h>
#include <stdlib.h>

#define GRAY_R 0.299f
#define GRAY_G 0.587f
#define GRAY_B 0.114f
#define PIXELS_TO_SHOW 5

static float rgb_to_gray(unsigned char r, unsigned char g, unsigned char b)
{
    return GRAY_R * r + GRAY_G * g + GRAY_B * b;
}

static float pixel_to_gray(const unsigned char *pixels, size_t pixel_index,
                           int channels)
{
    size_t offset = pixel_index * (size_t)channels;

    if (channels >= 3) {
        return rgb_to_gray(pixels[offset], pixels[offset + 1], pixels[offset + 2]);
    }

    return pixels[offset];
}

static void print_pixel(const unsigned char *pixels, size_t pixel_index,
                        int channels)
{
    size_t offset = pixel_index * (size_t)channels;
    float gray = pixel_to_gray(pixels, pixel_index, channels);

    if (channels == 1) {
        printf("Pixel [%zu]  gray=%-3u  -> gray=%.2f\n",
               pixel_index, pixels[offset], gray);
    } else if (channels == 2) {
        printf("Pixel [%zu]  gray=%-3u  A=%-3u  -> gray=%.2f\n",
               pixel_index, pixels[offset], pixels[offset + 1], gray);
    } else if (channels == 3) {
        printf("Pixel [%zu]  R=%-3u  G=%-3u  B=%-3u  -> gray=%.2f\n",
               pixel_index, pixels[offset], pixels[offset + 1], pixels[offset + 2],
               gray);
    } else {
        printf("Pixel [%zu]  R=%-3u  G=%-3u  B=%-3u  A=%-3u  -> gray=%.2f\n",
               pixel_index, pixels[offset], pixels[offset + 1], pixels[offset + 2],
               pixels[offset + 3], gray);
    }
}

int main(int argc, char *argv[])
{
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <image_path>\n", argv[0]);
        fprintf(stderr, "Example: %s ../dataset/raw/clase_0/PNG1.png\n",
                argv[0]);
        return EXIT_FAILURE;
    }

    const char *file_path = argv[1];
    int width = 0;
    int height = 0;
    int channels = 0;

    unsigned char *pixels = stbi_load(file_path, &width, &height, &channels, 0);
    if (pixels == NULL) {
        fprintf(stderr, "ERROR: could not load image '%s'\n", file_path);
        fprintf(stderr, "stb_image reason: %s\n", stbi_failure_reason());
        return EXIT_FAILURE;
    }

    size_t total_pixels = (size_t)width * (size_t)height;
    size_t pixels_to_show = total_pixels < PIXELS_TO_SHOW
                                ? total_pixels
                                : PIXELS_TO_SHOW;

    printf("\n=== Image Reader Test - Week 1 ===\n");
    printf("File     : %s\n", file_path);
    printf("Width    : %d px\n", width);
    printf("Height   : %d px\n", height);
    printf("Channels : %d  (1=gray, 2=gray+alpha, 3=RGB, 4=RGBA)\n",
           channels);
    printf("Total px : %zu\n", total_pixels);
    printf("-----------------------------------------\n");

    for (size_t i = 0; i < pixels_to_show; i++) {
        print_pixel(pixels, i, channels);
    }

    double sum_gray = 0.0;
    for (size_t i = 0; i < total_pixels; i++) {
        sum_gray += pixel_to_gray(pixels, i, channels);
    }

    double avg_gray = sum_gray / (double)total_pixels;
    double avg_gray_norm = avg_gray / 255.0;

    printf("-----------------------------------------\n");
    printf("Avg gray (raw)       : %.2f / 255\n", avg_gray);
    printf("Avg gray (normalized): %.4f\n", avg_gray_norm);

    stbi_image_free(pixels);
    printf("\nOK: image loaded and freed correctly.\n\n");

    return EXIT_SUCCESS;
}
