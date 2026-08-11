```mermaid
flowchart TD
    subgraph Input
        IN["Input Volume<br/>4 × 128 × 128 × 64<br/>(T1, T2, T1ce, FLAIR)"]
    end

    subgraph Encoder["Encoder Path"]
        E0["GraphSE-ResBlock₀<br/>4 → f₀"]
        E1["MaxPool3D(2) → GraphSE-ResBlock₁<br/>f₀ → f₁"]
        E2["MaxPool3D(2) → GraphSE-ResBlock₂<br/>f₁ → f₂"]
        E3["MaxPool3D(2) → GraphSE-ResBlock₃<br/>f₂ → f₃"]
    end

    subgraph Bottleneck
        BN["GraphSE-ResBlock_bn<br/>f₃ → 2·f₃"]
    end

    subgraph Decoder["Decoder Path"]
        D3["Upsample(2) ⊕ skip₃ → GraphSE-ResBlock<br/>2·f₃+f₃ → f₃"]
        D2["Upsample(2) ⊕ skip₂ → GraphSE-ResBlock<br/>f₃+f₂ → f₂"]
        D1["Upsample(2) ⊕ skip₁ → GraphSE-ResBlock<br/>f₂+f₁ → f₁"]
        D0["Upsample(2) ⊕ skip₀ → GraphSE-ResBlock<br/>f₁+f₀ → f₀"]
    end

    subgraph Output
        HEAD["Conv3D 1×1×1<br/>f₀ → 3"]
        OUT["Segmentation Map<br/>3 × 128 × 128 × 64<br/>(TC, WT, ET)"]
    end

    IN --> E0 --> E1 --> E2 --> E3
    E3 --> BN
    BN --> D3 --> D2 --> D1 --> D0
    D0 --> HEAD --> OUT

    E0 -. "skip₀" .-> D0
    E1 -. "skip₁" .-> D1
    E2 -. "skip₂" .-> D2
    E3 -. "skip₃" .-> D3
```

And the block-level detail for `GraphSE-ResBlock` itself, matching the fused design we settled on:

```mermaid
flowchart TD
    X["X (Cin)"] --> C1["Conv3D 3×3×3"] --> BN1["BN + ReLU"] --> C2["Conv3D 3×3×3"] --> BN2["BatchNorm"]
    BN2 --> H["H"]

    H --> Pool["Grid AvgPool<br/>V ← pool(H), stride s"]
    Pool --> GAT["Graph Attention<br/>single-head GAT — the 'squeeze'"]
    GAT --> Excite["Linear + Sigmoid<br/>g = σ(Linear(V')) — 'excite'"]
    Excite --> Interp["Interpolate<br/>g → H's resolution"]

    H --> Mult(("⊙"))
    Interp --> Mult
    Mult --> Add(("+"))

    X --> Shortcut["Shortcut 𝒫(X)<br/>1×1×1 Conv + BN<br/>(if Cin ≠ Cout, else identity)"]
    Shortcut --> Add

    Add --> ReLU2["ReLU"] --> Y["Y (Cout)"]
```

The two diagrams are consistent with each other — every `GraphSE-ResBlock` box in the top-level architecture expands to the second diagram. Only structural change from your earlier version: one block per level instead of two (`ResConvBlock` + `GraphSEBlock`), so the encoder/decoder path is now nine fused blocks instead of nine block-pairs.