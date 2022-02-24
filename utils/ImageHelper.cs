using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System;
using System.Numerics.Tensors;
using System.Numerics;

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

            // Preprocess image
            DenseTensor<float> input = new DenseTensor<float>(new[] { 1, 3, 224, 224 });
            var mean = new[] { 0.485f, 0.456f, 0.406f };
            var stddev = new[] { 0.229f, 0.224f, 0.225f };

            image.ProcessPixelRows(pixelAccessor =>
            {
                for (var y = 0; y < image.Height; y++)
                {
                    var pixelSpan = pixelAccessor.GetRowSpan(y);
                    for (int x = 0; x < image.Width; x++)
                    {
                        input[0, 0, y, x] = ((pixelSpan[x].R / 255f) - mean[0]) / stddev[0];
                        input[0, 1, y, x] = ((pixelSpan[x].G / 255f) - mean[1]) / stddev[1];
                        input[0, 2, y, x] = ((pixelSpan[x].B / 255f) - mean[2]) / stddev[2];
                    }
                }
            });


            // Faster conversion from RGB to Tensor?
            DenseTensor<float> input_faster = new DenseTensor<float>(new[] { 1, 3, 224, 224 });
            Memory<float> memory = input_faster.Buffer.Slice(0);
            //Span<float> span = memory.Span;
            var strides = input.Strides;
            float[] stridesRepacked = new float[] { strides[0], strides[1], strides[2], strides[3] };
            
            var strideVector = new Vector4(stridesRepacked);

            image.ProcessPixelRows(pixelAccessor =>
            {
                for (var y = 0; y < image.Height; y++)
                {
                    var pixelSpan = pixelAccessor.GetRowSpan(y);
                    for (int x = 0; x < image.Width; x++)
                    {
                        // Create the dot product of the coordinates to calculate the position of the pixel.
                        var indexArray = new float[4] { 0, 0, y, x};
                        var indexVector = new Vector4(indexArray);
                        var indexRed = (int)Vector4.Dot(strideVector, indexVector);

                        var indexArrayGreen = new float[4] { 0, 1, y, x};
                        var indexVectorGreen = new Vector4(indexArrayGreen);
                        var indexGreen = (int)Vector4.Dot(strideVector, indexVectorGreen);

                        var indexArrayBlue = new float[4] { 0, 2, y, x };
                        var indexVectorBlue = new Vector4(indexArrayBlue);
                        var indexBlue = (int)Vector4.Dot(strideVector, indexVectorBlue);

                        //Console.WriteLine($"{indexRed} {indexGreen} {indexBlue}");

                        memory.Span[indexRed] = ((pixelSpan[x].R / 255f) - mean[0]) / stddev[0];
                        memory.Span[indexGreen] = ((pixelSpan[x].G / 255f) - mean[1]) / stddev[1];
                        memory.Span[indexBlue] = ((pixelSpan[x].B / 255f) - mean[2]) / stddev[2];

                    }
                }
            });


            Microsoft.ML.OnnxRuntime.Tensors.Tensor<float> tensor = new Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<float>(memory, new[] { 1, 3, 224, 224 });
            return tensor;
        }
    }
}
