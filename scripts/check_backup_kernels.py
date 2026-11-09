"""Compare installed accelerated delta-rule kernels with the pure Torch reference."""
import inspect,json
import torch
from transformers.models.qwen3_5 import modeling_qwen3_5 as ops

torch.set_num_threads(8);torch.manual_seed(17)
for batch in [1,8]:
    shape=(batch,64,48,128)
    q,k,v=[torch.randn(shape,device='cuda',dtype=torch.bfloat16) for _ in range(3)]
    g=-torch.rand(shape[:3],device='cuda');beta=torch.rand(shape[:3],device='cuda',dtype=torch.bfloat16)
    with torch.inference_mode():
        kwargs={'g':g,'beta':beta,'output_final_state':True,'use_qk_l2norm_in_kernel':True}
        fast,state=ops.torch_chunk_gated_delta_rule(q,k,v,**kwargs)
        ref,refstate=inspect.unwrap(ops.torch_chunk_gated_delta_rule)(q,k,v,**kwargs)
        assert torch.isfinite(fast).all() and torch.isfinite(state).all()
        torch.testing.assert_close(fast,ref,atol=.002,rtol=.03)
        torch.testing.assert_close(state,refstate,atol=.01,rtol=.03)
        report={'batch':batch,'heads':48,'chunk_output_max_error':(fast-ref).abs().max().item(),'chunk_state_max_error':(state-refstate).abs().max().item()}
        last=[x[:,-1:].contiguous() for x in (q,k,v,g,beta)]
        kwargs={'g':last[3],'beta':last[4],'initial_state':state.clone(),'output_final_state':True,'use_qk_l2norm_in_kernel':True}
        fast,fs=ops.torch_recurrent_gated_delta_rule(*last[:3],**kwargs)
        kwargs['initial_state']=state.clone()
        ref,rs=inspect.unwrap(ops.torch_recurrent_gated_delta_rule)(*last[:3],**kwargs)
        torch.testing.assert_close(fast,ref,atol=.002,rtol=.03)
        torch.testing.assert_close(fs,rs,atol=.01,rtol=.03)
        report.update(recurrent_output_max_error=(fast-ref).abs().max().item(),recurrent_state_max_error=(fs-rs).abs().max().item(),passed=True)
        print(json.dumps(report),flush=True)
