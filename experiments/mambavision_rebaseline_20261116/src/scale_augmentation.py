"""Mild zoom-out composed with original FP32 affine geometry."""
import types,torch
def compose_scale(matrices,factors):
    assert matrices.ndim==3 and matrices.shape[1:]==(2,3)
    assert factors.shape==(len(matrices),)
    return matrices/factors[:,None,None]
def wrap_data(data,enabled=True):
    original_geometry=data.geometry
    def geometry(self,epoch):
        order,matrices=original_geometry(epoch)
        generator=torch.Generator().manual_seed(930042+epoch*100003)
        factors=.85+.15*torch.rand(len(self.ids['train']),generator=generator)
        self.scale_factors=factors.to(matrices.device)
        # affine_grid maps output to input, so size factor s uses inverse1/s.
        if enabled:matrices=compose_scale(matrices,self.scale_factors)
        return order,matrices
    data.geometry=types.MethodType(geometry,data)
    return data
