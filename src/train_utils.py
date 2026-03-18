"""
Train/eval loops
"""

import time
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from scipy.stats import pearsonr
from torch.cuda.amp import GradScaler, autocast
from torch.distributions import Normal
from torch.distributions import kl_divergence as kl
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb
from src.data_utils import densify, sparsify
from src.losses import (
    compute_laplacian,
    get_reconstruction_loss,
    graph_laplacian_regularization,
)


def _get_decoder_gene_weights(
    model: Optional[torch.nn.Module], out: Dict[str, Any]
) -> Optional[torch.Tensor]:
    """
    Resolve the decoder matrix mapping latent/metagene features to genes.
    """
    if "gene_weights" in out:
        return out["gene_weights"]

    if model is None:
        return None

    decoder = getattr(model, "metagenes_decoder", None)
    if decoder is None:
        return None

    if hasattr(decoder, "px_rate_decoder") and len(decoder.px_rate_decoder) > 0:
        return decoder.px_rate_decoder[0].weight
    if hasattr(decoder, "px_scale_decoder") and len(decoder.px_scale_decoder) > 0:
        return decoder.px_scale_decoder[0].weight
    return None


def train(
    config: Any,
    model: torch.nn.Module,
    loader: DataLoader,
    val_loader: Optional[DataLoader],
    epochs: Optional[int] = None,
    use_wandb: bool = True,
    **kwargs,
) -> None:
    """
    Trains the model
    :param config: Config object (e.g. Wandb config) with hyperparameters
    :param model: Model to train
    :param loader: Train loader
    :param val_loader: Validation loader
    :param use_wandb: whether to log the statistics into wandb
    :param kwargs: keyword arguments for the train and evaluate methods
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    runtime_kwargs = {**kwargs}
    runtime_kwargs.pop("device", None)
    laplacian_matrix = runtime_kwargs.get("laplacian_matrix", None)
    adjacency_matrix = runtime_kwargs.get("adjacency_matrix", None)
    if laplacian_matrix is None and adjacency_matrix is not None:
        laplacian_matrix = compute_laplacian(adjacency_matrix.to(device))
    if laplacian_matrix is not None:
        runtime_kwargs["laplacian_matrix"] = laplacian_matrix.to(device)

    optimiser = torch.optim.Adam(model.parameters(), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.99, patience=5, min_lr=0.00001
    )

    # Train/eval loop
    if epochs is None:
        epochs = config.epochs
    assert epochs is not None

    best_loss = float("inf")
    patience = 10
    no_improve = 0
    val_losses: Dict[str, float] = {}

    with tqdm(range(epochs), desc="Training") as pbar:
        for epoch in pbar:
            start_time = time.time()
            losses = train_step(
                model=model,
                optimiser=optimiser,
                loader=loader,
                beta=config.beta,
                device=device,
                **runtime_kwargs,
            )
            scheduler.step(losses["loss"])
            losses_dict = {f"train/{k}": v for k, v in losses.items()}
            losses_dict["train/epoch_time"] = time.time() - start_time

            if val_loader is not None:
                val_losses = eval_step(
                    model=model,
                    loader=val_loader,
                    beta=config.beta,
                    device=device,
                    **runtime_kwargs,
                )

                for k, v in val_losses.items():
                    losses_dict[f"val/{k}"] = v

                # Save best model
                if "loss" in val_losses:
                    if val_losses["loss"] < best_loss:
                        best_loss = val_losses["loss"]
                        torch.save(model.state_dict(), "data/best_model.pth")
                        no_improve = 0
                    else:
                        no_improve += 1

                    if no_improve >= patience:
                        print(f"Early stopping triggered at epoch {epoch + 1}")
                        break

            if use_wandb:
                wandb.log(losses_dict)

            # Cleaner tqdm display
            postfix = {"loss": f"{losses['loss']:.4f}"}
            if val_loader is not None and "loss" in val_losses:
                postfix["val_loss"] = f"{val_losses['loss']:.4f}"
                if "pearson" in val_losses:
                    postfix["pearson"] = f"{val_losses['pearson']:.4f}"
            pbar.set_postfix(postfix)


def train_step(
    model: torch.nn.Module,
    optimiser: torch.optim.Optimizer,
    loader: DataLoader,
    **kwargs,
) -> Dict[str, float]:
    """
    Performs one training step (i.e. one epoch)
    :param model: Model to train
    :param optimiser: Torch optimiser
    :param loader: Train loader
    :param kwargs: keyword arguments (currently unused)
    :return: epoch's train loss
    """
    model.train()
    losses_all = {}
    total_samples = 0
    step_kwargs = {**kwargs}
    device = step_kwargs.pop("device", next(model.parameters()).device)
    scaler = GradScaler(enabled=device.type == "cuda")
    total_grad_norm = 0.0
    valid_batches = 0

    for data in tqdm(loader, desc="Train batches", leave=False):
        if device is not None:
            data = data.to(device)

        optimiser.zero_grad(set_to_none=True)

        with autocast(enabled=device.type == "cuda"):
            out, node_features = forward(data, model, device=device, **step_kwargs)
            losses = compute_loss(data, out, node_features, model=model, **step_kwargs)

        loss = losses["loss"]

        if torch.isnan(loss):
            print("NaN loss detected, skipping batch")
            continue

        batch_size = data.x_target.shape[0] if hasattr(data, "x_target") else 1
        total_samples += batch_size

        scaler.scale(loss).backward()
        scaler.unscale_(optimiser)
        total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        total_grad_norm += total_norm.item()

        scaler.step(optimiser)
        scaler.update()
        valid_batches += 1

        for k, v in losses.items():
            v_item = v.item() if torch.is_tensor(v) else float(v)
            if k in losses_all:
                losses_all[k] += v_item * batch_size
            else:
                losses_all[k] = v_item * batch_size

    out_losses = (
        {k: v / total_samples for k, v in losses_all.items()}
        if total_samples > 0
        else {}
    )

    if valid_batches > 0:
        out_losses["grad_norm"] = total_grad_norm / valid_batches

    return out_losses


def eval_step(model: torch.nn.Module, loader: DataLoader, **kwargs) -> Dict[str, float]:
    """
    Performs evaluation step
    :param model: Model to evaluate
    :param optimiser: Torch optimiser
    :param loader: Validation loader
    :param kwargs: keyword arguments (currently unused)
    :return: losses
    """
    model.eval()
    losses_all = {}
    total_samples = 0
    eval_kwargs = {**kwargs}
    device = eval_kwargs.pop("device", next(model.parameters()).device)

    with torch.no_grad():
        for data in loader:
            if device is not None:
                data = data.to(device)

            out, node_features = forward(data, model, device=device, **eval_kwargs)
            losses = compute_loss(data, out, node_features, model=model, **eval_kwargs)
            metrics = compute_metrics(data, out, node_features, **eval_kwargs)
            losses = {**losses, **metrics}

            # Pearson correlation
            if hasattr(data, "x_target") and isinstance(out, dict) and "px_rate" in out:
                target = data.x_target.detach().cpu().numpy().flatten()
                pred = out["px_rate"].detach().cpu().numpy().flatten()
                if len(target) == len(pred):
                    corr, _ = pearsonr(pred, target)
                    losses["pearson"] = corr

            batch_size = data.x_target.shape[0] if hasattr(data, "x_target") else 1
            total_samples += batch_size

            for k, v in losses.items():
                v_item = v.item() if torch.is_tensor(v) else float(v)
                if k in losses_all:
                    losses_all[k] += v_item * batch_size
                else:
                    losses_all[k] = v_item * batch_size

    return (
        {k: v / total_samples for k, v in losses_all.items()}
        if total_samples > 0
        else {}
    )


def encode(
    data: Any,
    model: torch.nn.Module,
    preprocess_fn: Optional[Callable] = None,
    **kwargs,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Produces features of nodes in the hypergraph
    :param data: Data object to be fed to the model
    :param model: Hypergraph model
    :param preprocess_fn: Function that processes the input data
    :return: Node features of nodes appearing in data
    """
    x_source = data.x_source
    if preprocess_fn is not None:
        # Compute log1p (just for input data)
        x_source = preprocess_fn(data.x_source)

    # Prediction model
    x_source = model.encode_metagenes(x_source)

    # Sparsify data
    metagenes = model.metagenes
    hyperedge_index, hyperedge_attr = sparsify(data.source, metagenes, x=x_source)

    # Compute node features
    node_features = model(
        hyperedge_index, hyperedge_attr, dynamic_node_features=data.node_features
    )

    return node_features


def decode(
    data: Any,
    model: torch.nn.Module,
    node_features: Tuple[Dict[str, Any], Dict[str, Any]],
    use_observed_library: bool = True,
    n_cells: Optional[torch.Tensor] = None,
    library: Optional[torch.Tensor] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Decodes the target data according to data.target.
    :param data: Data object (only information about the target nodes is used).
                 data.target is a dictionary mapping node types to lists of node indices for that type
                 (i.e. similar to edge index in pytorch geometric, but with named node types)
    :param model: Hypergraph model
    :param node_features: Encoded node features
    :param use_observed_library: Whether to use observed library sizes or predict them
    :return: Dictionary with parameters of generative model
    """
    metagenes = model.metagenes
    target_hyperedge_index, _ = sparsify(data.target, metagenes, x=None)

    # Compute predictions for each metagene in the target tissues
    x_pred_metagenes = model.predict(
        target_hyperedge_index, node_features, **kwargs
    )  # Out shape=(nb_metagenes, metagene_dim)

    # Densify data
    x_pred_metagenes = densify(
        data.target, metagenes, target_hyperedge_index, x_pred_metagenes
    )

    # Factor that multiplies library size (i.e. number of cells in the summed signature). For deconvolution experiment,
    # we set this value (extrinsic to the model) to the number of cells of the summed signature at train time. This is
    # because averaging signatures result in "non-integer counts" and so NB/ZINB losses cannot be used. At test time,
    # we predict the "average" signatures (the number of cells in the signature is 1, i.e. n_cells=1)
    use_observed_n_cells = n_cells == 0
    if use_observed_n_cells:
        n_cells = data.target_misc["n_cells"][:, None]

    log_library = None
    if use_observed_library:
        if library is None:
            library = data.x_target.sum(dim=-1, keepdims=True)
        if n_cells is not None:
            library = library / data.target_misc["n_cells"][:, None]
        if library is not None:
            log_library = torch.log(library)  # [:, None]

    # Map metagene features back to high-dimensional space
    out = model.decode_metagenes(
        x_pred_metagenes, log_library=log_library, n_cells=n_cells, **kwargs
    )

    return out


def forward(
    data: Any,
    model: torch.nn.Module,
    device: Optional[torch.device] = None,
    preprocess_fn: Optional[Callable] = None,
    use_observed_library: bool = True,
    use_latent_means: bool = False,
    **kwargs,
) -> Tuple[Dict[str, Any], Tuple[Dict[str, Any], Dict[str, Any]]]:
    """
    Performs forward step on data
    :param data: Data to be fed to the model
    :param model: Pytorch model
    :param device: Pytorch device
    :param preprocess_fn: Function that processes the input data
    :param use_observed_library: Whether to use observed library sizes or predict them
    :param use_latent_means: Whether to use means of latent distribution (i.e. instead of sampling)
    :param kwargs: keyword arguments (currently unused)
    :return: predictions, loss, and node features (individual features, tissue features, metagene features)
    """
    # Compute node features
    node_features = encode(data, model, preprocess_fn=preprocess_fn)

    # Set latent variables to mean value (i.e. instead of sampling)
    if use_latent_means:
        (dynamic_node_features, static_node_features) = node_features
        for k in dynamic_node_features.keys():
            dynamic_node_features[k]["latent"] = dynamic_node_features[k]["mu"]
        node_features = (dynamic_node_features, static_node_features)

    # Decode
    out = decode(
        data, model, node_features, use_observed_library=use_observed_library, **kwargs
    )

    return out, node_features


def compute_loss(
    data: Any,
    out: Dict[str, Any],
    node_features: Tuple[Dict[str, Any], Dict[str, Any]],
    beta: float = 1.0,
    **kwargs,
) -> Dict[str, torch.Tensor]:
    """
    Computes VAE loss
    :param data: Data object with ground truth targets
    :param out: Output dict from the model
    :param node_features: Encoded node features
    :param beta: beta hyperparameter of beta-VAE
    :return: dictionary of losses
    """

    rec_loss = torch.mean(get_reconstruction_loss(data.x_target, **out))
    lambda_reg = float(kwargs.get("lambda_reg", 0.0))

    reg_loss = torch.tensor(0.0, device=rec_loss.device)
    if "reg_loss" in out:
        reg_loss = out["reg_loss"]
    else:
        laplacian_matrix = kwargs.get("laplacian_matrix", None)
        gene_weights = _get_decoder_gene_weights(kwargs.get("model", None), out)
        if lambda_reg > 0 and laplacian_matrix is not None and gene_weights is not None:
            reg_loss = graph_laplacian_regularization(gene_weights, laplacian_matrix)

    kl_loss = torch.tensor(0.0, device=rec_loss.device)
    dynamic_node_features, _ = node_features
    for k, v in dynamic_node_features.items():
        if "mu" in v:
            mu = v["mu"]
            if "logvar" in v:
                sigma = torch.exp(0.5 * v["logvar"])
            elif "sigma" in v:
                sigma = v["sigma"]
            elif "var" in v:
                sigma = torch.sqrt(v["var"] + 1e-8)
            elif "scale" in v:
                sigma = v["scale"]
            else:
                continue

            q = Normal(mu, sigma)
            p = Normal(torch.zeros_like(mu), torch.ones_like(sigma))
            kl_loss += torch.mean(torch.sum(kl(q, p), dim=-1))

    # Compute loss
    loss = rec_loss + beta * kl_loss + lambda_reg * reg_loss
    out_dict = {
        "loss": loss,
        "rec_loss": rec_loss,
        "kl_loss": kl_loss,
        "reg_loss": reg_loss,
    }

    return out_dict


def compute_metrics(
    data: Any,
    out: Dict[str, Any],
    node_features: Tuple[Dict[str, Any], Dict[str, Any]],
    metric_fns: Optional[list] = None,
    **kwargs,
) -> Dict[str, float]:
    out_dict = {}

    if metric_fns is not None:
        for metric_fn in metric_fns:
            out_dict[metric_fn.__name__] = metric_fn(
                data.x_target.detach().cpu().numpy(), out
            )

    return out_dict
