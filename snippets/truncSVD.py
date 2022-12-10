def truncSVD(model, q1):
    sd = model.state_dict()

    U0, S0, Vh0 = torch.linalg.svd(
        sd['linear_relu_stack.0.weight'], full_matrices=False)
    U2, S2, Vh2 = torch.linalg.svd(
        sd['linear_relu_stack.2.weight'], full_matrices=False)
    U4, S4, Vh4 = torch.linalg.svd(
        sd['linear_relu_stack.4.weight'], full_matrices=False)

    for j in range(15):
        if j > q1:
            S0[j] = 0
        if j > q1:
            S2[j] = 0
    for j in range(10):
        if j > q1:
            S4[j] = 0

    sd['linear_relu_stack.0.weight'] = torch.matmul(
        torch.matmul(U0, torch.diag(S0)), Vh0)
    sd['linear_relu_stack.2.weight'] = torch.matmul(
        torch.matmul(U2, torch.diag(S2)), Vh2)
    sd['linear_relu_stack.4.weight'] = torch.matmul(
        torch.matmul(U4, torch.diag(S4)), Vh4)

    model.load_state_dict(sd)
    return