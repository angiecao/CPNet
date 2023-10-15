_base_ = [
    '../_base_/models/pfnet_dcnedge_point.py', '../_base_/datasets/Vaihingen_overlap.py',
    '../_base_/default_runtime.py', '../_base_/schedules/myschedule_new.py'
]
train_dataloader = dict(batch_size=16, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)