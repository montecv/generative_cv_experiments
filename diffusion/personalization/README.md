# Stable Diffusion Personalization Experiments

I compare four approaches to identity personalization in Stable Diffusion:

- IP-Adapter FaceID
- Textual Inversion
- DreamBooth
- LoRA

The goal is to explore how much model adaptation is actually needed to preserve a target identity, and how each method behaves when an additional OpenPose constraint is introduced.

This is a qualitative study rather than a strict benchmark. I use the same identity dataset and similar generation scenarios, but the inference backbones and some generation settings are not identical across all methods.

## Experiments

| Method | Adaptation | Training steps | Training time | Main idea |
|---|---|---:|---:|---|
| IP-Adapter FaceID | None for the target identity | 0 | — | Reference face embedding conditions the generation |
| Textual Inversion | Text embedding | 10,000 | ~25 min | Identity is represented by learned token embeddings |
| DreamBooth | U-Net + text encoder | 2,000 | ~18 min | Model weights are fine-tuned for the target identity |
| LoRA | Low-rank adapters, rank 128 | 5,000 | ~35 min | Parameter-efficient adaptation of the diffusion model |

All trainable methods use 512×512 training resolution.

## Evaluation

I test each method in two scenarios:

**Identity generation** — I generate a portrait using the personalized identity representation.

**Pose-controlled generation** — I combine the personalized model with OpenPose ControlNet to test whether identity information survives an additional structural constraint.

I compare the results qualitatively in terms of identity preservation, pose adherence, visual quality, and adaptation cost.

## Results

| Method | Identity preservation | With pose control | Training cost | Observation |
|---|---|---|---|---|
| IP-Adapter FaceID | Strong | Strong | None | Best overall identity preservation in this experiment without identity-specific training |
| Textual Inversion | Good | Good | Low parameter count | Surprisingly effective given that only a small text-space representation is learned |
| DreamBooth | Strong | Good | Full fine-tuning | Preserves identity well, but the pose-controlled result becomes more stylized |
| LoRA | Moderate | Moderate | Highest in this run | Learns the broad appearance, but fine facial identity is less stable |

## Key observations

### IP-Adapter FaceID

IP-Adapter FaceID gives the strongest overall identity conditioning in this experiment without requiring identity-specific model training.

The reference embedding transfers facial characteristics effectively, and the identity remains recognizable after OpenPose conditioning. This makes it a particularly strong option when a reference image is available at inference time.

### Textual Inversion

Textual Inversion produces a clear improvement over the base generation despite learning only a small set of embedding vectors.

The identity is less exact than with FaceID conditioning, but it remains recognizable even in the pose-controlled experiment. The result shows how much identity information can be encoded without changing the diffusion model itself.

### DreamBooth

DreamBooth successfully adapts the model to the target identity and preserves the main facial characteristics in both experiments.

However, the additional training capacity does not automatically produce the strongest result. In the pose-controlled generation I observe stronger stylistic changes and less natural rendering than in the unconstrained portrait.

### LoRA

LoRA moves the base model toward the target identity while keeping the original checkpoint unchanged.

In my configuration, however, exact facial similarity is weaker than with IP-Adapter FaceID, Textual Inversion, or DreamBooth, especially after pose conditioning.

The training cost is also the highest in this particular experiment because I use a relatively high LoRA rank (`128`) and 5,000 optimization steps. This should not be interpreted as a general property of LoRA.

## Conclusion

The experiments show that increasing the amount of trainable model capacity does not automatically improve identity preservation.

IP-Adapter FaceID gives the strongest overall result in this setup while requiring no identity-specific training. Textual Inversion is surprisingly competitive considering how little it modifies, while DreamBooth provides strong personalization at the cost of full model adaptation.

LoRA is the least convincing configuration in this particular run: the high-rank adapter requires the longest training time while still losing more fine identity detail than the other methods.

The main trade-off is therefore not simply between model size and quality. The way identity information is introduced into the diffusion process can be at least as important as the number of trainable parameters.

## Notebooks

| Experiment | Notebook |
|---|---|
| IP-Adapter FaceID | [`ip_adapter_faceid.ipynb`](./ip_adapter_faceid.ipynb) |
| Textual Inversion | [`textual_inversion.ipynb`](./textual_inversion.ipynb) |
| DreamBooth | [`dreambooth.ipynb`](./dreambooth.ipynb) |
| LoRA | [`lora.ipynb`](./lora.ipynb) |

## Notes

The experiments are intended as a qualitative comparison rather than a controlled benchmark.

IP-Adapter FaceID and the Textual Inversion inference experiment use Realistic Vision V5.1, while DreamBooth and LoRA are based directly on Stable Diffusion v1.5. Textual Inversion embeddings are trained from the Stable Diffusion v1.5 checkpoint and then evaluated with the compatible Realistic Vision pipeline.

Because of these differences, the visual results should be interpreted as observations about the tested configurations rather than universal rankings of the personalization methods.