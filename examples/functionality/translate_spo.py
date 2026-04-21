"""Demonstrate cyclic physical-site translation on a sparse Pauli operator."""

import spd


if __name__ == "__main__":
    spd.numpy_backend.utils.set_packbit(32)
    spo = spd.numpy_backend.create_op(
        {
            "XYZIIZ": 1.0,
            "IYZZII": -0.5,
        }
    )

    translated = spo.translate(x=2, system_size=6)

    print("Original SPO:")
    print(spo)
    print()
    print("Translated SPO (right shift by 2 over 6 physical sites):")
    print(translated)
