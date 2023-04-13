from sbi_smfs.inference.generate_simulations import generate_simulations


def test_generate_simulations():
    observations = generate_simulations(
        "tests/test.config",
        num_sim=3,
        num_workers=1,
        file_name=None,
        show_progressbar=False,
        save_as_file=False,
    )
    assert observations[0].shape == (3, 13)
    assert observations[1].shape == (3, 2400)


def test_generate_simulations_with_Dx():
    observations = generate_simulations(
        "tests/test_2.config",
        num_sim=3,
        num_workers=1,
        file_name=None,
        show_progressbar=False,
        save_as_file=False,
    )
    assert observations[0].shape == (3, 14)
    assert observations[1].shape == (3, 2400)


test_generate_simulations()
