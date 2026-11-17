"""Deterministic mild RGB augmentation before frozen-prefix caching."""
import types,torch
def apply_color(images,brightness,contrast,saturation):
    # Inputs RGB [0,1]; each factor is shape N,1,1,1.
    x=(images*brightness).clamp(0,1)
    gray=(x*images.new_tensor([.2989,.587,.114])[None,:,None,None]).sum(1,keepdim=True)
    mean=gray.mean((2,3),keepdim=True)
    x=(contrast*x+(1-contrast)*mean).clamp(0,1)
    gray=(x*x.new_tensor([.2989,.587,.114])[None,:,None,None]).sum(1,keepdim=True)
    return (saturation*x+(1-saturation)*gray).clamp(0,1)

def wrap_data(data,enabled):
    old_geometry=data.geometry;old_batch=data.batch
    def geometry(self,epoch):
        result=old_geometry(epoch)
        generator=torch.Generator().manual_seed(920042+epoch*100003)
        n=len(self.ids['train'])
        self.color_factors=torch.stack([.85+.3*torch.rand(n,generator=generator),
            .85+.3*torch.rand(n,generator=generator),.9+.2*torch.rand(n,generator=generator)],1).to(self.images['train'].device)
        return result
    def batch(self,split,indices,matrices=None):
        images,target=old_batch(split,indices,matrices)
        if enabled and split=='train' and matrices is not None:
            # Disable autocast for augmentation, matching original FP32 geometry.
            with torch.autocast(images.device.type,enabled=False):
                rgb=images.float()*self.std+self.mean
                factors=self.color_factors[indices]
                rgb=apply_color(rgb,*[factors[:,i,None,None,None] for i in range(3)])
                images=(rgb-self.mean)/self.std
        return images,target
    data.geometry=types.MethodType(geometry,data)
    data.batch=types.MethodType(batch,data)
    return data
