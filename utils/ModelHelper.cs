using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System;
using System.Collections.Generic;
using System.Linq;

namespace OnnxRuntime.ResNet.Template.utils
{
    public static class ModelHelper
    {
        public static List<Prediction> GetPredictions(Tensor<float> input, string modelFilePath)
        {
            // Setup inputs
            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("data", input)
            };

            // Run inference
            var session = new InferenceSession(modelFilePath);
            var results = session.Run(inputs).First().AsEnumerable<float>();

            // Postprocess to get softmax vector
            float sum = results.Sum(x => (float)Math.Exp(x));
            List<float> softmax = results.Select(x => (float)Math.Exp(x) / sum).ToList();

            // Extract top 10 predicted classes
            List<Prediction> top10 = softmax.Select((x, i) => new Prediction { Label = LabelMap.Labels[i], Confidence = x })
                               .OrderByDescending(x => x.Confidence)
                               .Take(10).ToList();
            return top10;

        }
    }
}
