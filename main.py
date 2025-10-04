import datetime
import time
import pandas as pd
import torch
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
from itertools import chain
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from src.utils import getAccuracy
from src.model import WARVDLightning
from src.prepare import JoernExeError, GraphDataLightning, DatasetBuilder
from src.utils import PROJECT_ROOT, parse_args, process_joern_error, setup, extract_representations, ListToCSV, extract_train_set_repre, extract_validation_set_repre, extract_test_set_repre, save_pickle, load_pickle
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
import pickle as cPickle
from datetime import datetime
from src.utils import generate_id_labels

# run full model warvd

def main(args):

    # check if namespace is empty
    if not len(vars(args)):
        # setup was invoked
        setup()
        print('Setup is finished.')
        return 0
    test_build = args.scope == 'sample'
    if 'architecture' not in args:
        # prepare was invoked
        try:
            DatasetBuilder(fresh_build=True, test_build=test_build)
            return 0
        except JoernExeError:
            return process_joern_error(test_build)
    # model run was invoked
    model_kwargs = {
        'input_channels': 783, # 115-Word2Vec, 783-CodeBERT
        'hidden_channels': 168, # 240-Word2Vec, 168-CodeBERT
        'edge_attr_dim': 3, #3
        'num_layers': 4 #4, 6
    }
    lr = 0.0001
    data_kwargs = {
        'fresh_build': args.rebuild, 
        'test_build': test_build,
        'num_nodes': np.inf if args.architecture == 'flat' else 240,
        'train_proportion': 0.70,
        'batch_size': 16
    }
    pl.seed_everything(42)
    model = WARVDLightning(args.architecture, lr, **model_kwargs)
    torch.backends.cudnn.benchmark = True # Enable CUDA optimization
    try:
        data_module = GraphDataLightning(**data_kwargs) # Run this for the first time.

    except JoernExeError:
        return process_joern_error(test_build)
    # Lightning training
    current_time = datetime.now().strftime("%b%d_%H-%M-%S")
    checkpoint_callback = ModelCheckpoint(monitor="val_loss",
                                          verbose=True,
                                          mode='min',
                                          dirpath=os.getcwd() + os.sep + "data" + os.sep + "models" + os.sep + 'codebert' + os.sep,
                                          save_top_k = 8,
                                          filename='{epoch:02d}-{val_loss:.6f}-{val_acc:.6f}' + "_" + current_time + "codeBERT_783-168-3-4-BS16_realWorldDB")
    trainer = pl.Trainer(accelerator='gpu',
                         max_epochs= 12 if data_kwargs['test_build'] else 32,
                         log_every_n_steps = 6 if data_kwargs['test_build'] else 10, callbacks=[checkpoint_callback])

    train_dataloader = data_module.train_dataloader()
    #train_dataloader = data_module.sample_dataloader()
    val_dataloader = data_module.val_dataloader()
    test_dataloader = data_module.test_dataloader()

    if 'representations' not in args:
        trainer.fit(model, train_dataloader, val_dataloader)
        train_results = trainer.test(dataloaders = train_dataloader, verbose=True)
        print('TRAIN RESULTS')
        for metric, value in train_results[0].items():
            print('  ', metric.replace('test', 'train'))
            print('    ', value)
        test_results = trainer.test(dataloaders = test_dataloader, verbose=True)
        print('TEST RESULTS')
        for metric, value in test_results[0].items():
            print('  ', metric)
            print('    ', value)
        predicted_results = trainer.predict(model, dataloaders = test_dataloader)
        probs = []
        for batch in predicted_results:
            batch_probs = batch[0].tolist()  # 转换为Python列表
            probs.extend(batch_probs)
        accuracy, predicted_classes = getAccuracy(probs, data_module.test_y)
        print ('GNN_network' + " classification result: \n")
        print ("Total accuracy: " + str(accuracy))
        print ("----------------------------------------------------")
        print ("The confusion matrix: \n")
        target_names = ["Non-vulnerable", "Vulnerable"]  # non-vulnerable->0, vulnerable->1
        print(confusion_matrix(data_module.test_y, predicted_classes, labels=[0, 1]))
        print("\r\n")
        print(classification_report(data_module.test_y, predicted_classes, target_names=target_names))
        # Wrap the result to a CSV file.
        if not isinstance(probs, list): probs = probs.tolist()
        if not isinstance(data_module.test_set_id, list): test_id = np.asarray(data_module.test_set_id).tolist()
        zippedlist = list(zip(data_module.test_set_id, probs, data_module.test_y))
        result_set = pd.DataFrame(zippedlist, columns=['Function_ID', 'Probs. of being vulnerable', 'Label'])
        output_dir = './result' + os.sep
        ListToCSV(result_set, output_dir + os.sep + 'WAGGNN_CodeBERT_rw_v1' + '_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '_result.csv')

    else:
        # Extract representations
        #lg_model = model.load_from_checkpoint(checkpoint_path="E:\\Graph_representation\\devign-main\\data\\models\\epoch=02-val_loss=0.13-val_acc=0.00.ckpt")
        #ckpt_path = r'D:\Research\GGNN_test\data\models\word2vec\epoch=17-val_loss=0.069707-val_acc=0.000000_May18_19-44-06newModel_200_5_128_New2.ckpt' #word2vec
        #ckpt_path = r'D:\Research\GGNN_test\data\models\codebert\epoch=03-val_loss=0.040061-val_acc=0.000000_Jul04_22-41-23_FineTuned_CodeBERT_783_168_5_5x_BS16.ckpt' #Fine-tuned codeBERT
        #ckpt_path = r'D:\Research\GGNN_test\data\models\codebert\epoch=23-val_loss=0.066809-val_acc=0.000000_Jul08_22-11-25_CodeBERT_783_160_5_5x_BS16.ckpt'
        #ckpt_path = r'D:\Research\GGNN_test\data\models\codebert\epoch=25-val_loss=0.062480-val_acc=0.000000_Jul15_16-22-58_improvedModel_783-192-3-4-BS16.ckpt' #Improved attention model
        ckpt_path = r'D:\Research\GGNN_test\data\models\codebert\epoch=09-val_loss=0.362239-val_acc=0.000000_Sep12_11-07-46fine-tuned_CodeBERT_783-168-3-4-BS16_cpg_libtiff.ckpt' #Fine-tuned codeBERT,better result!
        lg_model = WARVDLightning.load_from_checkpoint(checkpoint_path=ckpt_path)
        lg_model = lg_model.cuda()
        lg_model.eval()
        feature_arr = []

        # Using the loaded model to predict.
        predicted_results = trainer.predict(lg_model, dataloaders=test_dataloader)
        probs = []
        for batch in predicted_results:
            # 假设batch是元组 (probabilities, labels, ...)
            batch_probs = batch[0].tolist()  # 转换为Python列表
            probs.extend(batch_probs)
        accuracy, predicted_classes = getAccuracy(probs, data_module.test_y)
        print('GNN_network' + " classification result: \n")
        print("Total accuracy: " + str(accuracy))
        print("----------------------------------------------------")
        print("The confusion matrix: \n")
        target_names = ["Non-vulnerable", "Vulnerable"]  # non-vulnerable->0, vulnerable->1
        print(confusion_matrix(data_module.test_y, predicted_classes, labels=[0, 1]))
        print("\r\n")
        print(classification_report(data_module.test_y, predicted_classes, target_names=target_names))
        # Wrap the result to a CSV file.

        if not isinstance(probs, list): probs = probs.tolist()
        if not isinstance(data_module.test_set_id, list): test_id = np.asarray(data_module.test_set_id).tolist()
        zippedlist = list(zip(data_module.test_set_id, probs, data_module.test_y))
        result_set = pd.DataFrame(zippedlist, columns=['Function_ID', 'Probs. of being vulnerable', 'Label'])
        output_dir = './result' + os.sep
        ListToCSV(result_set, output_dir + os.sep + 'GNN_Improved_network' + '_' + datetime.now().strftime(
            '%Y-%m-%d_%H-%M-%S') + '_result.csv')


        new_trainer = pl.Trainer(accelerator='gpu', callbacks=[checkpoint_callback])
        extract_train_set_repre(lg_model, new_trainer, data_module)
        extract_validation_set_repre(lg_model, new_trainer, data_module)
        extract_test_set_repre(lg_model, new_trainer, data_module)

        time.sleep(6)

        # Use LightGBM/XGBoost/Random Forest to train and test based on extracted representations.
        train_id = load_pickle("train_set_libtiff_id.pkl")
        train_repre = load_pickle("train_repre_Fine-tuned-codebert_libtiff.pkl")
        #train_repre = load_pickle("train_repre.pkl")
        train_y = load_pickle("train_set_libtiff_y.pkl")

        validation_id = load_pickle("validation_set_libtiff_id.pkl")
        validation_repre = load_pickle("validation_repre_Fine-tuned-codebert_libtiff.pkl")
        #validation_repre = load_pickle("validation_repre.pkl")
        validation_y = load_pickle("validation_set_libtiff_y.pkl")

        test_id = load_pickle("test_set_libtiff_id.pkl")
        test_repre = load_pickle("test_repre_Fine-tuned-codebert_libtiff.pkl")
        #test_repre = load_pickle("test_repre.pkl")
        test_y = load_pickle("test_set_libtiff_y.pkl")

        train_set_x = train_repre + validation_repre
        train_id = train_id
        train_set_y = train_y + validation_y
        """
        test_id = load_pickle("test_set_id.pkl")

        from src.utils import extract_features
        train_features, train_labels = extract_features(model, train_dataloader, 'cuda')
        val_features, val_labels = extract_features(model, val_dataloader, 'cuda')
        test_features, test_labels = extract_features(model, test_dataloader, 'cuda')

        train_set_x = train_features + val_features
        train_set_y = train_labels + val_labels
        test_repre = test_features
        test_y = test_labels
        """
        # print("---------------XGBoost------------------------------")
        # from src.model.xgboost_model import invokeXGBoost
        # invokeXGBoost(train_set_x, train_set_y, test_repre, test_y, test_id)
        # print("---------------Random Forest------------------------------")
        #from src.model.randomForest_model import invokeRandomForest
        #invokeRandomForest(train_set_x, train_set_y, test_repre, test_y, test_id)
        print("---------------LightGBM-------------------------------")
        from src.model.LightGBM import invokeLightGBM
        invokeLightGBM(train_set_x, train_set_y, test_repre, test_y, test_id)


    return 0

if __name__ == "__main__":
    args = parse_args()
    main(args)
