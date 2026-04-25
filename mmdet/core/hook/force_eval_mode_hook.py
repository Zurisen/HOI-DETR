from mmcv.runner import HOOKS, Hook

@HOOKS.register_module()
class ForceEvalModeHook(Hook):
    def before_train_epoch(self, runner):
        runner.model.eval()

    def before_train_iter(self, runner):
        runner.model.eval()