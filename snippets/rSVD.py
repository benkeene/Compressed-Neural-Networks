def rSVD(model, q):
    sd = model.state_dict()

    stack0 = torch.svd_lowrank(sd['linear_relu_stack.0.weight'], q)
    stack2 = torch.svd_lowrank(sd['linear_relu_stack.2.weight'], q)
    stack4 = torch.svd_lowrank(sd['linear_relu_stack.4.weight'], q)

    sd['linear_relu_stack.0.weight'] = (
        stack0[0] @ torch.diagflat(stack0[1]) @ torch.transpose(stack0[2], 0, 1))

    sd['linear_relu_stack.2.weight'] = (
        stack2[0] @ torch.diagflat(stack2[1]) @ torch.transpose(stack2[2], 0, 1))

    sd['linear_relu_stack.4.weight'] = (
        stack4[0] @ torch.diagflat(stack4[1]) @ torch.transpose(stack4[2], 0, 1))

    model.load_state_dict(sd)
    return