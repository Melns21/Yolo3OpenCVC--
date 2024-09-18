import onnx 
from onnxruntime.tools import optimize 
input_model_path = "E:/t4/onnxtestr/onnxtestr/yolov10n.onnx" 
output_model_path = "E:/t4/onnxtestr/onnxtestr/yolov10n_optimized.onnx" 
model = onnx.load(input_model_path) 
optimized_model = optimize.optimize_model(model) 
onnx.save_model(optimized_model, output_model_path) 
print(f"Optimized model saved to {output_model_path}") 
