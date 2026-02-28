from quick_model_loader.meta import ModelMeta

model_meta = ModelMeta.from_model_path("/home/nas_models/Qwen-72B-Chat/")

for sharding_meta in model_meta.sharding_metas:
    for tensors_meta in sharding_meta.tensors_metas:
        for tensor_name, tensor_info in tensors_meta.file_tensor_info_map.items():
            print(tensor_name, tensor_info.dtype)
