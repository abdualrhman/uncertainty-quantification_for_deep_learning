from cp import *


if __name__ == "__main__":
    # Fix randomness
    seed = 0
    np.random.seed(seed=seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    cache_fname = "./.cache/cifar10_aug_df.csv"
    alpha_table = 0.1
    try:
        df = pd.read_csv(cache_fname)
    except:
        # Configure experiment
        # modelnames = ['Cifar10ConvModel','ResNet152','ResNet101','ResNet50','ResNet18','DenseNet161','VGG16','Inception','ShuffleNet']
        modelnames = ['Cifar10ConvModel', 'Cifar10Resnet20']
        alphas = [0.05, 0.10]
        predictors = ['Fixed', 'Naive', 'APS', 'RAPS']
        params = list(itertools.product(modelnames, alphas, predictors))
        m = len(params)
        datasetname = 'Cifar10Aug'
        datasetpath = '/scratch/group/ilsvrc/val/'
        num_trials = 10
        kreg = None
        lamda = None
        randomized = True
        n_data_conf = 5000
        n_data_val = 5000
        pct_paramtune = 0.33
        bsz = 32
        cudnn.benchmark = True

        # Perform the experiment
        df = pd.DataFrame(
            columns=["Model", "Predictor", "Top1", "Top5", "alpha", "Coverage", "Size"])
        for i in range(m):
            modelname, alpha, predictor = params[i]
            print(
                f'Model: {modelname} | Desired coverage: {1-alpha} | Predictor: {predictor}')

            out = experiment(modelname, datasetname, datasetpath, num_trials,
                             params[i][1], kreg, lamda, randomized, n_data_conf, n_data_val, pct_paramtune, bsz, predictor)
            df = df.append({"Model": modelname,
                            "Predictor": predictor,
                            "Top1": np.round(out[0], 3),
                            "Top5": np.round(out[1], 3),
                            "F1": np.round(out[2], 3),
                            "alpha": alpha,
                            "Coverage": np.round(out[3], 3),
                            "Size":
                            np.round(out[4], 3)}, ignore_index=True)
        df.to_csv(cache_fname)
    # Print the TeX table
    table_str = make_table(df, alpha_table)
    table = open(
        f"outputs/cifar10_aug_results_{alpha_table}".replace('.', '_') + ".tex", 'w')
    table.write(table_str)
    table.close()
