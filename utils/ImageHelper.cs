using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System;
using System.Numerics.Tensors;

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

            Memory<float> memory = input.Buffer.Slice(0);

            Microsoft.ML.OnnxRuntime.Tensors.Tensor<float> tensor = new Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<float>(memory, new[] { 1, 3, 224, 224 });
            return tensor;
        }
    }
}
