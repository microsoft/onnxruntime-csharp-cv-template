using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System;
using System.Numerics.Tensors;
using System.Numerics;
using System.Collections.Generic;

namespace OnnxRuntime.ResNet.Template
{
    public static class ImageHelper
    {
        public static Microsoft.ML.OnnxRuntime.Tensors.Tensor<float> GetImageTensorFromPath(string imageFilePath, int imgWidth = 224, int imgHeight=224)
        {
            // Read image
            using Image<Rgb24> image = Image.Load<Rgb24>(imageFilePath);

            // Resize image
            image.Mutate(x =>
            {
                x.Resize(new ResizeOptions
                {
                    Size = new Size(imgWidth, imgHeight),
                    Mode = ResizeMode.Crop
                });
            });

            var images = new List<Image<Rgb24>>() {image};
            var mean = new[] { 0.485f, 0.456f, 0.406f };
            var stddev = new[] { 0.229f, 0.224f, 0.225f };

            // Original Method
            var memory = RGBToTensorOriginal(images, mean, stddev);


            // Faster conversion from RGB to Tensor?
            var memory = RGBToTensorDirectEdit(images, mean, stddev);


            Microsoft.ML.OnnxRuntime.Tensors.Tensor<float> tensor = new Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<float>(memory, new[] { 1, 3, 224, 224 });
            return tensor;
        }

        private static Memory<float> RGBToTensorOriginal(List<Image<Rgb24>> images, float[] mean, float[] stddev)
        {
            DenseTensor<float> input = new DenseTensor<float>(new[] { images.Count, 3, 224, 224});

            for (var i=0; i < images.Count; i++)
            {
                images[i].ProcessPixelRows(pixelAccessor =>
                {
                    for (var y = 0; y < images[i].Height; y++)
                    {
                        var pixelSpan = pixelAccessor.GetRowSpan(y);
                        for (int x = 0; x < images[i].Width; x++)
                        {
                            input[i, 0, y, x] = ((pixelSpan[x].R / 255f) - mean[0]) / stddev[0];
                            input[i, 1, y, x] = ((pixelSpan[x].G / 255f) - mean[1]) / stddev[1];
                            input[i, 2, y, x] = ((pixelSpan[x].B / 255f) - mean[2]) / stddev[2];
                        }
                    }
                });
            }

            var memory = input.Buffer.Slice(0);
            return memory;
        }


        private static Memory<float> RGBToTensorDirectEdit(List<Image<Rgb24>> images, float[] mean, float[] stddev)
        {
            DenseTensor<float> input_faster = new DenseTensor<float>(new[] {images.Count, 3, 224, 224});
            Memory<float> memory = input_faster.Buffer.Slice(0);
            var strides = input_faster.Strides;
            float[] stridesRepacked = new float[] {strides[0], strides[1], strides[2], strides[3]};

            var strideVector = new Vector4(stridesRepacked);
            for (var i = 0; i < images.Count; i++)
            {
                images[i].ProcessPixelRows(pixelAccessor =>
                {
                    for (var y = 0; y < images[i].Height; y++)
                    {
                        var pixelSpan = pixelAccessor.GetRowSpan(y);
                        for (int x = 0; x < images[i].Width; x++)
                        {
                            // Create the dot product of the coordinates to calculate the position of the pixel.
                            var indexArray = new float[4] {i, 0, y, x};
                            var indexVector = new Vector4(indexArray);
                            var indexRed = (int) Vector4.Dot(strideVector, indexVector);

                            var indexArrayGreen = new float[4] {i, 1, y, x};
                            var indexVectorGreen = new Vector4(indexArrayGreen);
                            var indexGreen = (int) Vector4.Dot(strideVector, indexVectorGreen);

                            var indexArrayBlue = new float[4] {i, 2, y, x};
                            var indexVectorBlue = new Vector4(indexArrayBlue);
                            var indexBlue = (int) Vector4.Dot(strideVector, indexVectorBlue);

                            //Console.WriteLine($"{indexRed} {indexGreen} {indexBlue}");

                            memory.Span[indexRed] = ((pixelSpan[x].R / 255f) - mean[0]) / stddev[0];
                            memory.Span[indexGreen] = ((pixelSpan[x].G / 255f) - mean[1]) / stddev[1];
                            memory.Span[indexBlue] = ((pixelSpan[x].B / 255f) - mean[2]) / stddev[2];
                        }
                    }
                });
            }
            return memory;
        }
    }
}
