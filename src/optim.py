def parameter_groups(model, lr_backbone, lr_head):
    """Only the two new embedding heads receive the head learning rate."""
    head_params = [p for module in (model.image_encoder.proj, model.text_encoder.proj)
                   for p in module.parameters() if p.requires_grad]
    head_ids = {id(p) for p in head_params}
    backbone_params = [p for p in model.parameters()
                       if p.requires_grad and id(p) not in head_ids]
    return [
        {'params': backbone_params, 'lr': lr_backbone},
        {'params': head_params, 'lr': lr_head},
    ]
