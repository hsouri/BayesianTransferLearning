from torch.utils.data import DataLoader


def get_tests_final(test_loader, test_datasets, num_workers, batch_size):
    test_loaders = {'test': test_loader}
    for test_datasets_name in test_datasets:
        test_datasets_inner = test_datasets[test_datasets_name]
        loader = DataLoader(test_datasets_inner, batch_size=batch_size, num_workers=num_workers)
        test_loaders[test_datasets_name] = loader
    return test_loaders
