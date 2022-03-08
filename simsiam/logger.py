import warnings

from torch.utils.tensorboard import SummaryWriter


class TBLogger(SummaryWriter):
    def __init__(self, log_dir=None, comment='', purge_step=None, max_queue=10, flush_secs=120, filename_suffix='',
                 global_step=0, batch_size=1, world_size=1, global_step_divider=1):
        super().__init__(log_dir, comment, purge_step, max_queue, flush_secs, filename_suffix)
        self.global_step = global_step
        self.warned_missing_grad = False
        self.batch_size = batch_size
        self.world_size = world_size
        self.dist_bs = self.batch_size * self.world_size
        self.global_step_divider = global_step_divider

    def add_scalar(self, tag, scalar_value, global_step=None, walltime=None):
        if global_step is None:
            global_step = round(self.global_step / self.global_step_divider)
        super().add_scalar(tag, scalar_value, global_step, walltime)

    def add_histogram(self, tag, values, global_step=None, bins='tensorflow', walltime=None, max_bins=None):
        if global_step is None:
            global_step = round(self.global_step / self.global_step_divider)
        super().add_histogram(tag, values, global_step, bins, walltime, max_bins)

    def add_image(self, tag, img_tensor, global_step=None, walltime=None, dataformats='CHW'):
        if global_step is None:
            global_step = round(self.global_step / self.global_step_divider)
        super().add_image(tag, img_tensor, global_step, walltime, dataformats)

    def add_figure(self, tag, figure, global_step=None, close=True, walltime=None):
        if global_step is None:
            global_step = round(self.global_step / self.global_step_divider)
        super().add_figure(tag, figure, global_step, close, walltime)

    def add_text(self, tag, text_string, global_step=None, walltime=None):
        super().add_text(tag, text_string, global_step, walltime)

    def log_describe(self, name, tensor, global_step=None):
        self.add_scalar(f"{name}_mean", tensor.mean(), global_step=global_step)
        self.add_scalar(f"{name}_std", tensor.std(), global_step=global_step)
        self.add_histogram(f"{name}", tensor, global_step=global_step)

    def describe_model(self, model):
        global_step = round(self.global_step / self.global_step_divider)
        for name, param in model.named_parameters():

            self.add_histogram(name, param.clone().cpu().data.numpy(), global_step=global_step)
            self.add_scalar(f"model_describe/{name}_mean", param.clone().cpu().data.numpy().mean())
            self.add_scalar(f"model_describe/{name}_std", param.clone().cpu().data.numpy().std())
            try:
                self.add_histogram(name + "_grad", param.grad.clone().cpu().data.numpy(),
                                          global_step=global_step)
                self.add_scalar(f"model_describe/{name}_grad_mean", param.grad.clone().cpu().data.numpy().mean())
                self.add_scalar(f"model_describe/{name}_grad_std", param.grad.clone().cpu().data.numpy().std())
            except (ValueError, AttributeError):
                if not self.warned_missing_grad:
                    warnings.warn(name + " has wrong gradient value")
                    self.warned_missing_grad = True

    def describe_model_step(self, params_before_step, params_after_step):
        global_step = round(self.global_step / self.global_step_divider)
        for (b_name, b_param), (name, param) in zip(params_before_step.items(), params_after_step.items()):
            try:
                self.add_histogram(name + "_step", (b_param - param),
                                   global_step=global_step)
                self.add_scalar(f"model_describe/{name}_step_mean", (b_param - param).mean())
                self.add_scalar(f"model_describe/{name}_step_std", (b_param - param).std())
            except (ValueError, AttributeError):
                if not self.warned_missing_grad:
                    warnings.warn(name + " has wrong gradient value")
                    self.warned_missing_grad = True

    def log_config(self, args):
        config_text = ""
        for group_name, variable_dict in args.items():
            config_text += f"{group_name}:  \n"
            for name, value in variable_dict.items():
                config_text += f"--{name} : {value}  \n"
        self.add_text("config", config_text)
        
    def step(self):
        self.global_step += self.dist_bs

    def need_log(self, per_step):
        assert per_step >= self.dist_bs, \
            f"per_step ({per_step}) < distributed batch size ({self.dist_bs})"
        assert per_step >= self.global_step_divider, \
            f"per_step ({per_step}) < global_step_divider ({self.global_step_divider})"
        return (round(self.global_step / self.dist_bs) % round(per_step / self.dist_bs)) == 0