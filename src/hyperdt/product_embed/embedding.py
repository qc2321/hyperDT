import torch

from geoopt import ManifoldParameter


def _mse(y, y_pred):
    return torch.mean((y - y_pred) ** 2)


def embed(dists, space, burn_in_epochs=100, training_epochs=1000, lr=3e-4, print_freq=100):
    # Input validation: distance matrix
    dists = torch.Tensor(dists)
    assert torch.allclose(dists, dists.T)
    assert (dists >= 0).all()
    n = dists.shape[0]
    idx = torch.triu_indices(row=n, col=n, offset=1)
    target_dists = dists[idx]
    print(f"Embedding {n} points in {space.dims} dimensions")

    # Randomly initialize points
    # TODO: maybe seed with MDS instead
    x_embed = torch.cat(
        [component.embed_point(0.1 * torch.rand(n, component.dim)) for component in space.components], axis=1
    )
    x_embed = torch.nan_to_num(x_embed, 0)
    # x_embed.requires_grad = True
    x_embed = ManifoldParameter(x_embed, manifold=space)

    lr = lr / 1000

    # opt = torch.optim.Adam([x_embed], lr=lr)
    opt = 
    for epoch in range(burn_in_epochs + training_epochs):
        opt.zero_grad()
        if epoch == burn_in_epochs:
            lr *= 1000

        achieved_dists = space.dist(x_embed, x_embed)[idx]
        loss = _mse(target_dists, achieved_dists)
        # loss.backward()

        # Project back to manifold - TODO: do this better
        # x_embed = torch.cat([c.embed_point(x[:, 1:]) for c, x in zip(space.components, space.split(x_embed))], axis=1)

        # Use Riemannian gradient descent to update all coordinates
        loss.backward()
        with torch.no_grad():
            for component in space.components:
                x_embed[:, : component.dim + 1] = component.embed_point(x_embed[:, : component.dim + 1] - lr * x_embed.grad)

        if epoch % print_freq == 0:
            print(f"Epoch {epoch}:\tLoss:{loss}")

    return x_embed
