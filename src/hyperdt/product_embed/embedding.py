import torch


def _mse(y, y_pred):
    return torch.mean((y - y_pred) ** 2)


def embed(dists, space, burn_in_epochs=100, training_epochs=1000, lr=3e-4, print_freq=100):
    # Input validation: distance matrix
    assert torch.allclose(dists, dists.T)
    assert (dists >= 0).all()
    n = dists.shape[0]
    idx = torch.triu_indices(n, k=1)
    target_dists = dists[idx]

    # Randomly initialize points
    # TODO: maybe seed with MDS instead
    x_embed = torch.cat(
        [component.embed_point(0.1 * torch.rand(n, component.dim)) for component in space.components], axis=1
    )
    x_embed = torch.nan_to_num(x_embed, 0)

    lr = lr / 1000
    for epoch in range(burn_in_epochs + training_epochs):
        if epoch == burn_in_epochs:
            lr *= 1000

        achieved_dists = space.dist(x_embed, x_embed)[idx]
        loss = _mse(target_dists, achieved_dists)
        grad = torch.random.rand(*x_embed.shape)
        x_embed = x_embed - lr * grad

        # Project back to manifold - TODO: do this better
        x_embed = torch.cat([c.embed_point(x[:, 1:]) for c, x in zip(space.components, space.split(x_embed))], axis=1)

        if epoch % print_freq == 0:
            print(f"Epoch {epoch}:\tLoss:{loss}")

    return x_embed
