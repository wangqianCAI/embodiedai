import os
"""Finish an incomplete gradient-accumulation group before each epoch checkpoint."""
class EpochAccumulation:
    def __init__(self, epoch_microsteps, accumulation_steps, initial_microstep=0):
        self.epoch_microsteps = epoch_microsteps
        self.accumulation_steps = accumulation_steps
        self.completed = initial_microstep
        self.tail = epoch_microsteps % accumulation_steps

    def before(self, accelerator):
        position = self.completed % self.epoch_microsteps
        if self.tail and position == self.epoch_microsteps - self.tail:
            assert accelerator.step % self.accumulation_steps == 0
            accelerator.gradient_accumulation_steps = self.tail

    def after(self, accelerator):
        self.completed += 1
        if self.completed % self.epoch_microsteps == 0:
            assert accelerator.sync_gradients, 'Epoch checkpoint must follow an optimizer update'
            accelerator.gradient_accumulation_steps = self.accumulation_steps
            accelerator.step = 0
