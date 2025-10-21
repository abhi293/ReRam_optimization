import torch
print(torch.version.cuda)           # Should print 12.1
print(torch.cuda.is_available())    # Should print True
print(torch.cuda.get_device_name(0)) # Should print your RTX GPU name 