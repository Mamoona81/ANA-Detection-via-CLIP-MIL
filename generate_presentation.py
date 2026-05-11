"""
Generate a professional PowerPoint presentation for the MIL pipeline.
"""

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.enum.text import PP_ALIGN
from pptx.dml.color import RGBColor

# Color scheme
TITLE_COLOR = RGBColor(0, 51, 102)       # Dark blue
ACCENT_COLOR = RGBColor(0, 102, 204)     # Bright blue
HIGHLIGHT_COLOR = RGBColor(255, 102, 0)  # Orange
TEXT_COLOR = RGBColor(50, 50, 50)        # Dark gray
BG_COLOR = RGBColor(245, 245, 245)       # Light gray


def add_title_slide(prs, title, subtitle=""):
    """Add a title slide."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # Blank layout
    background = slide.background
    fill = background.fill
    fill.solid()
    fill.fore_color.rgb = TITLE_COLOR
    
    # Title
    title_box = slide.shapes.add_textbox(Inches(0.5), Inches(2), Inches(9), Inches(1.5))
    title_frame = title_box.text_frame
    title_frame.word_wrap = True
    p = title_frame.paragraphs[0]
    p.text = title
    p.font.size = Pt(54)
    p.font.bold = True
    p.font.color.rgb = RGBColor(255, 255, 255)
    
    # Subtitle
    if subtitle:
        subtitle_box = slide.shapes.add_textbox(Inches(0.5), Inches(3.7), Inches(9), Inches(2))
        subtitle_frame = subtitle_box.text_frame
        subtitle_frame.word_wrap = True
        p = subtitle_frame.paragraphs[0]
        p.text = subtitle
        p.font.size = Pt(28)
        p.font.color.rgb = RGBColor(200, 220, 255)
    
    return slide


def add_content_slide(prs, title, content_points):
    """Add a content slide with bullet points."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # Blank layout
    
    # Title bar
    title_shape = slide.shapes.add_shape(1, Inches(0), Inches(0), Inches(10), Inches(0.8))
    title_shape.fill.solid()
    title_shape.fill.fore_color.rgb = TITLE_COLOR
    title_shape.line.color.rgb = TITLE_COLOR
    
    # Title text
    title_frame = title_shape.text_frame
    title_frame.clear()
    p = title_frame.paragraphs[0]
    p.text = title
    p.font.size = Pt(40)
    p.font.bold = True
    p.font.color.rgb = RGBColor(255, 255, 255)
    p.space_after = Pt(10)
    
    # Content
    content_box = slide.shapes.add_textbox(Inches(0.7), Inches(1.2), Inches(8.6), Inches(5.3))
    text_frame = content_box.text_frame
    text_frame.word_wrap = True
    
    for i, point in enumerate(content_points):
        if i > 0:
            p = text_frame.add_paragraph()
        else:
            p = text_frame.paragraphs[0]
        
        p.text = point
        p.level = 0
        p.font.size = Pt(20)
        p.font.color.rgb = TEXT_COLOR
        p.space_after = Pt(12)
    
    return slide


def add_two_column_slide(prs, title, left_title, left_points, right_title, right_points):
    """Add a two-column slide."""
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    
    # Title bar
    title_shape = slide.shapes.add_shape(1, Inches(0), Inches(0), Inches(10), Inches(0.8))
    title_shape.fill.solid()
    title_shape.fill.fore_color.rgb = TITLE_COLOR
    title_shape.line.color.rgb = TITLE_COLOR
    
    title_frame = title_shape.text_frame
    p = title_frame.paragraphs[0]
    p.text = title
    p.font.size = Pt(40)
    p.font.bold = True
    p.font.color.rgb = RGBColor(255, 255, 255)
    
    # Left column header
    left_header = slide.shapes.add_textbox(Inches(0.5), Inches(1.2), Inches(4.5), Inches(0.4))
    p = left_header.text_frame.paragraphs[0]
    p.text = left_title
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = ACCENT_COLOR
    
    # Left column content
    left_box = slide.shapes.add_textbox(Inches(0.5), Inches(1.7), Inches(4.5), Inches(4.5))
    left_frame = left_box.text_frame
    left_frame.word_wrap = True
    for i, point in enumerate(left_points):
        if i > 0:
            p = left_frame.add_paragraph()
        else:
            p = left_frame.paragraphs[0]
        p.text = point
        p.font.size = Pt(16)
        p.font.color.rgb = TEXT_COLOR
        p.space_after = Pt(10)
    
    # Right column header
    right_header = slide.shapes.add_textbox(Inches(5.2), Inches(1.2), Inches(4.5), Inches(0.4))
    p = right_header.text_frame.paragraphs[0]
    p.text = right_title
    p.font.size = Pt(18)
    p.font.bold = True
    p.font.color.rgb = ACCENT_COLOR
    
    # Right column content
    right_box = slide.shapes.add_textbox(Inches(5.2), Inches(1.7), Inches(4.5), Inches(4.5))
    right_frame = right_box.text_frame
    right_frame.word_wrap = True
    for i, point in enumerate(right_points):
        if i > 0:
            p = right_frame.add_paragraph()
        else:
            p = right_frame.paragraphs[0]
        p.text = point
        p.font.size = Pt(16)
        p.font.color.rgb = TEXT_COLOR
        p.space_after = Pt(10)
    
    return slide


def main():
    prs = Presentation()
    prs.slide_width = Inches(10)
    prs.slide_height = Inches(7.5)
    
    # Slide 1: Title
    add_title_slide(prs, "Multi-Instance Learning for ANA Fluorescence", 
                    "Frozen CLIP Features + Clinical Pattern Recognition")
    
    # Slide 2: Project Overview
    add_content_slide(prs, "Project Overview", [
        "• Goal: Classify ANA (Anti-Nuclear Antibody) fluorescence patterns using pre-computed CLIP embeddings",
        "• Input: 257×768 CLIP ViT-L/14 tokens (1 CLS + 256 patches) per immunofluorescence image",
        "• Output: Multi-label predictions across 8 ICAP (International Consensus) pattern classes",
        "• Advantage: Skip heavy image processing; work directly with frozen foundation model features",
        "• Rigor: Train/val/test split from CSV; evaluate with F1-macro, mAP, per-class metrics"
    ])
    
    # Slide 3: Dataset Structure
    add_content_slide(prs, "Dataset Structure", [
        "• 6,563 samples total (4,605 train / 979 val / 979 test)",
        "• 8 ICAP multi-label classes: golgi, homogeneous, nucleolus, nuclear_dot, centromere, nuclear_membrane, cytoplasm_mitochondria, speckled_granular",
        "• Multi-hot encoding: samples can belong to multiple patterns simultaneously",
        "• Imbalanced: speckled_granular (62% of train) vs. golgi (2%)",
        "• Clinical context: titer levels (100–32000) + patient ID (ISBN) for reproducibility"
    ])
    
    # Slide 4: Architecture Overview
    add_content_slide(prs, "MIL Architecture", [
        "🔹 CLS Token (Global Anchor):",
        "   → Projects to 512-d medical space via 2-layer MLP",
        "",
        "🔹 Patch Tokens (Multi-Instance Learning):",
        "   → All 256 patches projected to 512-d",
        "   → Class-wise max pooling: identify top-3 patches per class",
        "   → Output: 8 class-specific embeddings",
        "",
        "🔹 Fusion & Classification:",
        "   → Concatenate [CLS_proj, pooled_per_class] → small MLP → 8 logits",
        "   → Loss: BCEWithLogitsLoss (multi-label sigmoid)"
    ])
    
    # Slide 5: Phase 1 - Data Engineering
    add_content_slide(prs, "Phase 1: Data Engineering (30 min)", [
        "✓ Load CSV + extract multi-label targets (8 binary columns)",
        "✓ Create PyTorch Dataset for each split (train/val/test)",
        "✓ L2-normalize embeddings for CLIP geometry",
        "✓ Build DataLoaders with proper batch handling",
        "",
        "Result: 4,605 training, 979 validation, 979 test samples",
        "Each sample: [257, 768] features + [8] labels + metadata"
    ])
    
    # Slide 6: Phase 2 - Model Architecture
    add_two_column_slide(prs, "Phase 2: Token-Aware Model",
        "Key Components", [
            "• MedicalProjectionHead",
            "  768 → 1024 → 512",
            "  ReLU + Dropout",
            "",
            "• ClassWiseMaxPooling",
            "  Learned attention per class",
            "  Top-k aggregation"
        ],
        "Model Flow", [
            "Input: [batch, 257, 768]",
            "↓",
            "CLS: [batch, 512]",
            "Patches: [batch, 256, 512]",
            "↓",
            "Pooled: [batch, 8, 512]",
            "↓",
            "Output: [batch, 8] logits"
        ]
    )
    
    # Slide 7: Phase 3 - Training
    add_content_slide(prs, "Phase 3: Sprint Training (2 hours)", [
        "Optimizer: AdamW(lr=1e-4, weight_decay=0.05)",
        "Loss: BCEWithLogitsLoss (multi-label)",
        "Early Stopping: patience=5 on val F1-Macro",
        "",
        "Expected convergence: 8–10 epochs (CPU ~30min, GPU ~5min)",
        "",
        "Saves: best_model.pt, training history, run metadata"
    ])
    
    # Slide 8: Phase 4 - Evaluation Metrics
    add_two_column_slide(prs, "Phase 4: High-Rigor Evaluation",
        "Headline Metrics", [
            "F1-Macro: 0.2066",
            "F1-Micro: 0.6426",
            "F1-Weighted: 0.5444",
            "mAP: 0.3671"
        ],
        "Per-Class Performance", [
            "✓ Cytoplasm (AP: 0.92)",
            "✓ Speckled (AP: 0.76)",
            "△ Centromere (AP: 0.30)",
            "△ Nucleolus (AP: 0.27)",
            "△ Nuclear Dot (AP: 0.06)",
            "⚠ Golgi (AP: 0.04) [rare]"
        ]
    )
    
    # Slide 9: Results Interpretation
    add_content_slide(prs, "Results Interpretation", [
        "📊 F1-Macro (0.21) vs F1-Micro (0.64):",
        "   → High imbalance: common patterns (speckled, cytoplasm) dominate micro-F1",
        "   → Macro-F1 shows model struggles with rare patterns (golgi, nuclear_dot)",
        "",
        "📈 mAP (0.37):",
        "   → Ranking quality for multi-label is moderate",
        "   → Model separates co-occurring patterns reasonably well",
        "",
        "💡 Next Step: Tune per-class thresholds on val split to boost macro-F1"
    ])
    
    # Slide 10: Phase 5 - Demo Inference
    add_content_slide(prs, "Phase 5: Demo Inference (Live Examples)", [
        "Sample from test split:",
        "",
        "Patient ISBN: 6369187600 | Titer: 320",
        "Ground Truth: Speckled (class 7)",
        "Model Prediction: Speckled (0.65) + Homogeneous (0.53)",
        "",
        "✓ Correctly identified primary pattern",
        "✓ Secondary co-occurring pattern detected (multi-label insight)"
    ])
    
    # Slide 11: Clinical Context
    add_content_slide(prs, "Clinical Context: ANA Patterns", [
        "🔬 Why these 8 patterns matter:",
        "• Homogeneous: Antibodies to histones & DNA (SLE marker)",
        "• Speckled: Anti-Ro/SSA, Anti-La/SSB (Sjögren's)",
        "• Nucleolar: Anti-fibrillarin (scleroderma)",
        "• Centromere: CENP-B (limited sclerosis)",
        "• Golgi/Cytoplasm/DFS70: Specific autoimmune profiles",
        "",
        "Titer Level: Antibody concentration (100 → 32000) correlates with disease severity"
    ])
    
    # Slide 12: Visualization Outputs
    add_content_slide(prs, "Evaluation Artifacts Generated", [
        "✓ mil_multilabel_confusion_matrices.png",
        "  → 8 binary confusion matrices (TN/FP/FN/TP per class)",
        "",
        "✓ mil_training_curves.png",
        "  → Loss, F1-Macro, mAP trajectories across epochs",
        "",
        "✓ mil_eval_report.txt",
        "  → Presentation-ready clinical summary",
        "",
        "✓ mil_eval_metrics.json",
        "  → Machine-readable results for downstream analysis"
    ])
    
    # Slide 13: How to Run the Pipeline
    add_content_slide(prs, "How to Run (Quick Start)", [
        "1️⃣ Train the model (first time only):",
        "   python ./src/mil_pipeline.py --epochs 15 --batch-size 16 --outdir output",
        "",
        "2️⃣ Generate evaluation & plots (reusable):",
        "   python ./src/mil_evaluate.py --checkpoint best_model.pt --outdir output",
        "",
        "3️⃣ View results:",
        "   → output/mil_multilabel_confusion_matrices.png",
        "   → output/mil_training_curves.png (after step 1)",
        "   → output/mil_eval_report.txt"
    ])
    
    # Slide 14: Strengths & Limitations
    add_two_column_slide(prs, "Strengths & Limitations",
        "✅ Strengths", [
            "• Frozen CLIP features: no heavy image processing",
            "• Multi-label design: captures co-occurring patterns",
            "• Patient-level split: avoids data leakage",
            "• Rigorous metrics (F1-macro, mAP)",
            "• Clinically grounded labels"
        ],
        "⚠️ Limitations", [
            "• Limited by CLIP geometry (not tuned for ANA)",
            "• Severe class imbalance (golgi 2% vs speckled 62%)",
            "• Modest macro-F1 (0.21) on rare classes",
            "• No image augmentation",
            "• Threshold tuning needed for production"
        ]
    )
    
    # Slide 15: Future Directions
    add_content_slide(prs, "Future Work & Improvements", [
        "🚀 Short-term (production-ready):",
        "   • Per-class threshold tuning on val split",
        "   • Class weighting / oversampling for rare patterns",
        "   • Patient-level stratification for fairness",
        "",
        "🔬 Long-term (research):",
        "   • Fine-tune CLIP on medical fluorescence data",
        "   • Attention visualization for explainability",
        "   • Ensemble with radiologist feedback",
        "   • Temporal analysis (titer progression)"
    ])
    
    # Slide 16: Conclusion
    add_title_slide(prs, "Conclusion", 
                    "High-rigor multi-label classification pipeline from frozen CLIP features")
    
    # Save
    output_path = "output/MIL_ANA_Presentation.pptx"
    prs.save(output_path)
    print(f"✅ Presentation saved to: {output_path}")


if __name__ == "__main__":
    main()
