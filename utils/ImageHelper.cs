using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Diagnostics;

namespace OnnxRuntime.ResNet.Template
{
    public static class ImageHelper
    {
        /// <summary>
        /// Gets the image, and creates a tensor ready for use in OnnxRuntime
        /// </summary>
        /// <param name="imageFilePath"></param>
        /// <param name="imgWidth"></param>
        /// <param name="imgHeight"></param>
        /// <returns></returns>
        public static List<DenseTensor<float>> GetImageTensorFromPath(string imageFilePath, int imgWidth = 224, int imgHeight=224)
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

            // Shape of the input tensor to the OnnxRuntime model
            var inputDimensions = new int[] {1, 3, 224, 224};

            var tensors = ImageToTensor(images, mean, stddev, inputDimensions);

            return tensors;
        }

        /// <summary>
        /// Converts the list of images into batches and list of input tensors.
        /// </summary>
        /// <param name="images"></param>
        /// <param name="mean"></param>
        /// <param name="stddev"></param>
        /// <param name="inputDimension">The size of the tensor that the OnnxRuntime model is expecting [1, 3, 224, 224] </param>
        /// <returns></returns>
        private static List<DenseTensor<float>> ImageToTensor(List<Image<Rgb24>> images, float[] mean, float[] stddev, int[] inputDimension)
        {
            // Used to create more than one batch
            int numberBatches = 1;

            // If required, can create batches of different sizes
            var batchSizes = new int[] {images.Count};

            // Keep track of which tile we are using to create multiple batches.
            int tileIndex = 0;

            var strides = GetStrides(inputDimension);

            var inputs = new List<DenseTensor<float>>();

            for (var j = 0; j < numberBatches; j++)
            {

                inputDimension[0] = batchSizes[j];

                // Need to directly use a DenseTensor here because we need access to the underlying span.
                DenseTensor<float> input = new DenseTensor<float>(inputDimension);

                // This is the index used to get the stride offsets for the span.
                // mdIndex[0] = row in the batch.
                // mdIndex[1] = either 1/2/3 depending on if its for RBG
                // mdIndex[2] = y value corresponding to the current image height
                // mdIndex[3] = Always 0 since we want the start index of the x values
                int[] mdIndex = new int[4];
                mdIndex[3] = 0;
                for (var i = 0; i < batchSizes[j]; i++)
                {
                    mdIndex[0] = i;
                    var image = images[tileIndex];
                    image.ProcessPixelRows(pixelAccessor =>
                    {
                        var inputSpan = input.Buffer.Span;
                        for (var y = 0; y < image.Height; y++)
                        {
                            mdIndex[2] = y;

                            var rowSpan = pixelAccessor.GetRowSpan(y);

                            // Update the mdIndex based on R/G/B and get a span to the underlying memory
                            mdIndex[1] = 0;
                            var spanR = inputSpan.Slice(GetIndex(strides, mdIndex), image.Width);
                            mdIndex[1] = 1;
                            var spanG = inputSpan.Slice(GetIndex(strides, mdIndex), image.Width);
                            mdIndex[1] = 2;
                            var spanB = inputSpan.Slice(GetIndex(strides, mdIndex), image.Width);

                            // Now we can just directly loop through and copy the values directly from span to span.
                            for (int x = 0; x < image.Width; x++)
                            {
                                spanR[x] = ((rowSpan[x].R / 255f) - mean[0]) / stddev[0];
                                spanG[x] = ((rowSpan[x].G / 255f) - mean[1]) / stddev[1];
                                spanB[x] = ((rowSpan[x].B / 255f) - mean[2]) / stddev[2];
                            }
                        }
                    });

                    tileIndex++;
                    inputs.Add(input);
                }
            }

            return inputs;
        }


        /// <summary>
        /// Gets the set of strides that can be used to calculate the offset of n-dimensions in a 1-dimensional layout
        /// </summary>
        /// <param name="dimensions"></param>
        /// <param name="reverseStride"></param>
        /// <returns></returns>
        public static int[] GetStrides(ReadOnlySpan<int> dimensions, bool reverseStride = false)
        {
            int[] strides = new int[dimensions.Length];

            if (dimensions.Length == 0)
            {
                return strides;
            }

            int stride = 1;
            if (reverseStride)
            {
                for (int i = 0; i < strides.Length; i++)
                {
                    strides[i] = stride;
                    stride *= dimensions[i];
                }
            }
            else
            {
                for (int i = strides.Length - 1; i >= 0; i--)
                {
                    strides[i] = stride;
                    stride *= dimensions[i];
                }
            }

            return strides;
        }


        /// <summary>
        /// Calculates the 1-d index for n-d indices in layout specified by strides.
        /// </summary>
        /// <param name="strides"></param>
        /// <param name="indices"></param>
        /// <param name="startFromDimension"></param>
        /// <returns></returns>
        public static int GetIndex(int[] strides, ReadOnlySpan<int> indices, int startFromDimension = 0)
        {
            Debug.Assert(strides.Length == indices.Length);

            int index = 0;
            for (int i = startFromDimension; i < indices.Length; i++)
            {
                index += strides[i] * indices[i];
            }

            return index;
        }
    }
}
