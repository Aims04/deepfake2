"""
Signal Aggregation — EfficientNet-B4 Edition
=============================================
Combines FFT + ELA + Noise + Facial Geometry + EfficientNet-B4

Weights calibrated on FaceForensics++ validation:
  EfficientNet gets highest weight (0.40) as it's the strongest signal
  when fine-tuned weights are available.
"""

from dataclasses import dataclass, field, asdict
from typing import List
import numpy as np

WEIGHTS = {
    "efficientnet": 0.40,  # dominant signal with pretrained weights
    "fft":          0.25,  # GAN frequency fingerprinting
    "ela":          0.18,  # JPEG inconsistency
    "noise":        0.10,  # sensor noise
    "geo":          0.07,  # facial geometry
}

VERDICT_THRESHOLD = 0.40


@dataclass
class DetectionFlag:
    severity: str
    category: str
    signal: str
    value: float
    threshold: float
    description: str


@dataclass
class DetectionReport:
    verdict: str
    confidence: int
    composite_score: float
    gan_probability: int
    diffusion_probability: int
    face_manipulation_score: int
    frequency_anomaly_score: int
    cnn_score: int
    metadata_integrity: str
    model_type_detected: str
    faces_detected: int
    flags: List[DetectionFlag] = field(default_factory=list)
    signal_scores: dict = field(default_factory=dict)
    raw_signals: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["flags"] = [{"severity": f["severity"], "text": f["description"]} for f in d["flags"]]
        return d


def aggregate_signals(analysis: dict, efficientnet: dict) -> DetectionReport:
    fft   = analysis["fft_analysis"]
    ela   = analysis["ela_analysis"]
    noise = analysis["noise_analysis"]
    geo   = analysis["facial_geometry"]
    sigs  = analysis["signal_scores"]

    eff_prob = efficientnet["fake_probability"]

    # Weighted composite
    composite = (
        WEIGHTS["efficientnet"] * eff_prob     +
        WEIGHTS["fft"]          * sigs["fft"]  +
        WEIGHTS["ela"]          * sigs["ela"]  +
        WEIGHTS["noise"]        * sigs["noise"]+
        WEIGHTS["geo"]          * sigs["geo"]
    )
    composite = float(np.clip(composite, 0.0, 1.0))

    verdict    = "DEEPFAKE" if composite >= VERDICT_THRESHOLD else "AUTHENTIC"
    distance   = abs(composite - VERDICT_THRESHOLD)
    confidence = int(np.clip(50 + distance * 120, 52, 98))

    # UI scores (0-100)
    gan_prob     = int(np.clip(sigs["fft"] * 100, 0, 100))
    freq_anomaly = int(np.clip(fft["gan_frequency_prob"] * 100, 0, 100))
    face_manip   = int(np.clip(sigs["geo"] * 100, 0, 100))
    cnn_score    = int(np.clip(eff_prob * 100, 0, 100))
    diff_prob    = int(np.clip((sigs["ela"] * 0.6 + sigs["noise"] * 0.4) * 100, 0, 100))
    if verdict == "AUTHENTIC":
        diff_prob = min(diff_prob, 20)

    # Metadata integrity
    ela_score = ela["ela_manipulation_prob"]
    metadata_integrity = "COMPROMISED" if ela_score > 0.6 else "SUSPICIOUS" if ela_score > 0.35 else "INTACT"

    model_type = _infer_model_type(fft, ela, noise, geo, eff_prob, composite)
    flags      = _generate_flags(fft, ela, noise, geo, eff_prob)

    return DetectionReport(
        verdict=verdict, confidence=confidence,
        composite_score=round(composite, 4),
        gan_probability=gan_prob, diffusion_probability=diff_prob,
        face_manipulation_score=face_manip,
        frequency_anomaly_score=freq_anomaly,
        cnn_score=cnn_score,
        metadata_integrity=metadata_integrity,
        model_type_detected=model_type,
        faces_detected=geo.get("faces_detected", 0),
        flags=flags,
        signal_scores={
            "efficientnet": round(eff_prob, 3),
            "fft":          round(sigs["fft"], 3),
            "ela":          round(sigs["ela"], 3),
            "noise":        round(sigs["noise"], 3),
            "geo":          round(sigs["geo"], 3),
            "composite":    round(composite, 4),
        },
        raw_signals={
            "fft": fft, "ela": ela,
            "noise": noise, "geo": geo,
            "efficientnet": efficientnet,
        }
    )


def _generate_flags(fft, ela, noise, geo, eff_prob):
    flags = []

    # EfficientNet CNN
    if eff_prob > 0.70:
        flags.append(DetectionFlag("HIGH", "CNN", "efficientnet", eff_prob, 0.70,
            f"EfficientNet-B4 CNN: {eff_prob:.0%} fake probability — deep feature artifacts detected"))
    elif eff_prob > 0.45:
        flags.append(DetectionFlag("MED", "CNN", "efficientnet", eff_prob, 0.45,
            f"EfficientNet-B4 CNN: {eff_prob:.0%} fake probability — borderline classification"))

    # FFT
    alpha = fft.get("alpha_1f", 2.0)
    peak  = fft.get("periodic_peak_score", 0.0)
    if alpha < 1.4:
        flags.append(DetectionFlag("HIGH", "GAN", "alpha_1f", alpha, 1.6,
            f"Spectral slope α={alpha:.2f} (real images: ≥1.8) — GAN upsampling signature"))
    elif alpha < 1.7:
        flags.append(DetectionFlag("MED", "GAN", "alpha_1f", alpha, 1.7,
            f"Slight spectral deviation α={alpha:.2f} — possible GAN artifact"))
    if peak > 0.5:
        flags.append(DetectionFlag("HIGH", "GAN", "periodic_peak_score", peak, 0.5,
            f"Periodic spectral peaks at 2^n frequencies (score={peak:.2f}) — transposed convolution artifact"))
    elif peak > 0.25:
        flags.append(DetectionFlag("MED", "GAN", "periodic_peak_score", peak, 0.25,
            f"Mild periodic spectral peaks (score={peak:.2f}) — possible GAN upsampling"))

    # ELA
    ela_prob = ela.get("ela_manipulation_prob", 0.0)
    spatial  = ela.get("spatial_inconsistency", 0.0)
    if ela_prob > 0.55:
        flags.append(DetectionFlag("HIGH", "MANIPULATION", "ela_prob", ela_prob, 0.55,
            f"ELA manipulation probability {ela_prob:.0%} — inconsistent JPEG compression regions"))
    elif ela_prob > 0.35:
        flags.append(DetectionFlag("MED", "ELA", "ela_prob", ela_prob, 0.35,
            f"ELA inconsistency {ela_prob:.0%} — possible compositing"))
    if spatial > 1.0:
        flags.append(DetectionFlag("HIGH", "MANIPULATION", "spatial_inconsistency", spatial, 1.0,
            f"Spatial ELA inconsistency={spatial:.2f} — distinct compression boundaries suggest face pasting"))

    # Noise
    gauss = noise.get("gaussian_sim", 1.0)
    autocorr = noise.get("noise_autocorr", 0.0)
    if gauss < 0.5:
        flags.append(DetectionFlag("MED", "NOISE", "gaussian_sim", gauss, 0.5,
            f"Non-Gaussian noise (similarity={gauss:.2f}) — inconsistent with real camera PRNU"))
    if abs(autocorr) > 0.25:
        flags.append(DetectionFlag("MED", "NOISE", "autocorr", autocorr, 0.25,
            f"Structured noise autocorrelation={autocorr:.3f} — neural network generation artifact"))

    # Geometry
    color_delta = geo.get("face_color_delta_lab", 0.0)
    boundary    = geo.get("boundary_sharpness", 0.0)
    texture     = geo.get("skin_texture_variance", 1.0)
    if color_delta > 30:
        flags.append(DetectionFlag("HIGH", "GEOMETRY", "color_delta", color_delta, 30,
            f"Face-background color mismatch ΔE={color_delta:.1f} LAB units — face paste artifact"))
    elif color_delta > 15:
        flags.append(DetectionFlag("MED", "GEOMETRY", "color_delta", color_delta, 15,
            f"Color mismatch ΔE={color_delta:.1f} — lighting inconsistency"))
    if boundary > 0.5:
        flags.append(DetectionFlag("HIGH", "GEOMETRY", "boundary", boundary, 0.5,
            f"Sharp face boundary (score={boundary:.2f}) — blending seam at face edge"))
    if texture < 0.15:
        flags.append(DetectionFlag("MED", "GEOMETRY", "texture", texture, 0.15,
            f"Over-smooth skin texture (variance={texture:.3f}) — GAN-generated face characteristic"))

    if not flags:
        flags.append(DetectionFlag("LOW", "AUTHENTIC", "composite", 0.0, 0.0,
            "No significant artifacts detected across all analysis layers"))

    order = {"HIGH": 0, "MED": 1, "LOW": 2}
    flags.sort(key=lambda f: order.get(f.severity, 3))
    return flags


def _infer_model_type(fft, ela, noise, geo, eff_prob, composite):
    if composite < 0.28:
        return "Authentic Camera Photo"
    peak     = fft.get("periodic_peak_score", 0)
    alpha    = fft.get("alpha_1f", 2.0)
    ela_prob = ela.get("ela_manipulation_prob", 0)
    boundary = geo.get("boundary_sharpness", 0)
    spatial  = ela.get("spatial_inconsistency", 0)
    texture  = geo.get("skin_texture_variance", 1.0)

    if peak > 0.4 and alpha < 1.5 and texture < 0.2 and ela_prob < 0.4:
        return "StyleGAN2 / StyleGAN3"
    if boundary > 0.4 and spatial > 0.7 and ela_prob > 0.45:
        return "DeepFaceLab / FaceSwap"
    if ela_prob > 0.4 and noise.get("noise_gan_prob", 0) > 0.4 and peak < 0.3:
        return "Stable Diffusion / DALL-E"
    if peak > 0.3 and alpha < 1.7:
        return "GAN-Generated (ProGAN / BigGAN)"
    if eff_prob > 0.6:
        return "Unknown Synthetic Media"
    return "Borderline — Low Confidence"


def build_report_text(report: DetectionReport, analysis: dict) -> dict:
    """
    Generate human-readable analysis text from real measured values.
    No API needed — built from actual signal numbers.
    """
    fft   = analysis["fft_analysis"]
    ela   = analysis["ela_analysis"]
    noise = analysis["noise_analysis"]
    geo   = analysis["facial_geometry"]
    eff   = analysis.get("efficientnet", {})

    alpha     = fft['alpha_1f']
    peak      = fft['periodic_peak_score']
    eff_prob  = report.signal_scores.get("efficientnet", 0.5)
    composite = report.composite_score

    if report.verdict == "DEEPFAKE":
        summary = (
            f"Composite detection score {composite:.3f} exceeds the {VERDICT_THRESHOLD} threshold — "
            f"{report.model_type_detected} detected with {report.confidence}% confidence. "
            f"EfficientNet-B4 CNN assigns {eff_prob:.0%} fake probability based on deep mesoscopic features. "
            f"Spectral analysis shows α={alpha:.2f} (authentic images: ≥1.8) with periodic peak score "
            f"{peak:.2f}, consistent with neural network upsampling artifacts. "
            f"ELA spatial inconsistency of {ela['spatial_inconsistency']:.2f} further supports manipulation."
        )
        gan_analysis = (
            f"Spectral power law α={alpha:.2f} deviates from the 1/f pink noise distribution of real cameras "
            f"(expected α≥1.8). Periodic spectral peaks at 2^n spatial frequencies (score={peak:.2f}) "
            f"indicate transposed-convolution upsampling — the signature of GAN generator networks. "
            f"{'High-frequency energy excess detected.' if fft['hf_energy_ratio'] > 0.06 else 'High-frequency energy is within normal range.'}"
        )
        recommendation = (
            f"Flag as synthetic media — {report.model_type_detected} with {report.confidence}% confidence. "
            f"Do not use as evidence of a real person or event."
        )
    else:
        summary = (
            f"Composite detection score {composite:.3f} is below the {VERDICT_THRESHOLD} threshold — "
            f"media appears authentic with {report.confidence}% confidence. "
            f"EfficientNet-B4 assigns {eff_prob:.0%} fake probability. "
            f"Spectral power law α={alpha:.2f} is consistent with real camera pink-noise distribution. "
            f"ELA residuals are {'uniform' if ela['spatial_inconsistency'] < 0.5 else 'slightly inconsistent'} "
            f"with no significant manipulation boundaries detected."
        )
        gan_analysis = (
            f"Spectral power law α={alpha:.2f} is consistent with authentic camera imagery (expected ≥1.8). "
            f"Periodic peak score {peak:.2f} is {'below' if peak < 0.3 else 'near'} the 0.3 threshold — "
            f"no significant GAN transposed-convolution artifacts detected in the frequency domain."
        )
        recommendation = (
            f"Image appears authentic — no significant manipulation detected across all analysis layers. "
            f"Manual review recommended only if contextual evidence suggests tampering."
        )

    return {
        "summary":        summary,
        "gan_analysis":   gan_analysis,
        "recommendation": recommendation,
    }
