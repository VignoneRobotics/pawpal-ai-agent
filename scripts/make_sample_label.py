"""Generate the clearly FAKE prescription labels used in the PawPal+ demo (assets/sample_label*.png).

Fakeprofen is a made-up drug and the clinic is fictional. Run: python scripts/make_sample_label.py
"""

from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ASSETS = Path(__file__).resolve().parent.parent / "assets"
WIDTH, HEIGHT = 1200, 800
RED = (200, 30, 30)


def font(size: int) -> ImageFont.ImageFont:
    """Return Pillow's built-in scalable font at the given size."""
    return ImageFont.load_default(size=size)


def make_label(dose: str) -> Image.Image:
    """Draw one fake label with the given dose text."""
    img = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, WIDTH - 10, HEIGHT - 10], outline="black", width=4)
    draw.rectangle([10, 10, WIDTH - 10, 110], fill=RED)
    draw.text((WIDTH // 2, 60), "SAMPLE - NOT A REAL PRESCRIPTION", fill="white", font=font(52), anchor="mm")

    lines = [
        ("Example Vet Clinic (fictional)", 34),
        ("Rx # 000000        Date: 01/05/2026", 30),
        ("", 16),
        ("Patient: Mochi (dog)", 42),
        ("Drug: Fakeprofen (fictional) chewable tablets", 42),
        (f"Dose: {dose}", 42),
        ("Directions: give twice daily (every 12 hours)", 42),
        ("Prescriber: Dr. Example (fictional)", 30),
    ]
    y = 150
    for text, size in lines:
        draw.text((50, y), text, fill="black", font=font(size))
        y += size + 22

    draw.text((WIDTH // 2, HEIGHT - 50), "Demo label for PawPal+ testing only. Fakeprofen is not a real drug.",
              fill=RED, font=font(28), anchor="mm")

    # Light diagonal watermark (kept faint so the label stays readable).
    mark = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    ImageDraw.Draw(mark).text((WIDTH // 2, HEIGHT // 2), "SAMPLE", fill=(200, 30, 30, 40), font=font(260), anchor="mm")
    mark = mark.rotate(20, center=(WIDTH // 2, HEIGHT // 2))
    return Image.alpha_composite(img.convert("RGBA"), mark).convert("RGB")


def main() -> None:
    """Write the matching (25 mg) and mismatching (50 mg) sample labels."""
    ASSETS.mkdir(exist_ok=True)
    make_label("25 mg (1 tablet)").save(ASSETS / "sample_label.png")
    make_label("50 mg (2 tablets)").save(ASSETS / "sample_label_mismatch.png")
    print(f"Wrote {ASSETS / 'sample_label.png'} and {ASSETS / 'sample_label_mismatch.png'}")


if __name__ == "__main__":
    main()
