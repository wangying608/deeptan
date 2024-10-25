import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.utils.data.dataloader as Dataloader
import torch.utils.data.dataset as Dateset
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
import random
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def PCC(X,Y):
    xBar = np.mean(X)
    yBar = np.mean(Y)
    SSR = 0
    varX = 0
    varY = 0
    for i in range(0,len(X)):
        diffXXBar = X[i] - xBar
        diffYYBar = Y[i] - yBar
        SSR += (diffXXBar * diffYYBar)
        varX += diffXXBar ** 2
        varY += diffYYBar ** 2
    SST = math.sqrt(varX * varY)
    return SSR / SST

def omics_load(omics_dir: list):
    # omics_dir = []  #input omics dir
    output_omics_dir = []
    for dir in omics_dir:
        omics = pd.read_csv(dir)
        output_omics_dir.append(omics)

    all_omics = pd.concat(output_omics_dir, axis=1)
    all_omics = all_omics.to_numpy().astype(float)

    return all_omics

def MCAR(mask_rate, input_matrix):
    """
    Missing complete at random
    """

    row_num = input_matrix.shape[0]
    col_num = input_matrix.shape[1]
    mcar_mask = np.random.rand(row_num, col_num) < mask_rate
    mcar_matrix = input_matrix.mask(mcar_mask)

    masked_values = input_matrix.values[mcar_mask]

    # masked position
    row_pos = mcar_matrix.index[np.where(np.isnan(mcar_matrix))[0]]
    col_pos = mcar_matrix.columns[np.where(np.isnan(mcar_matrix))[1]]

    masked_position = [(i,j) for i, j in zip(row_pos, col_pos)]

    return mcar_matrix, masked_values, masked_position
    # return mcar_matrix, masked_values, row_pos, col_pos

def MNAR(mask_rate, input_matrix):
    """
    Missing not at random
    """
    mask_count = int(input_matrix.shape[0] * mask_rate)
    mnar_matrix = input_matrix.copy()
    masked_values = []

    for col in mnar_matrix.columns:
        # add nan value
        min_indices = mnar_matrix[col].nsmallest(mask_count).index
        masked_values.extend(mnar_matrix.loc[min_indices, col].tolist())
        mnar_matrix.loc[min_indices, col] = np.nan

    # masked position
    row_pos = mnar_matrix.index[np.where(np.isnan(mnar_matrix))[0]]
    col_pos = mnar_matrix.columns[np.where(np.isnan(mnar_matrix))[1]]

    masked_position = [(i, j) for i, j in zip(row_pos, col_pos)]

    return mnar_matrix, masked_values, masked_position
    # return mnar_matrix, masked_values, row_pos, col_pos

class dataset(Dateset.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = torch.Tensor(self.data[index])
        return data

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, 400)
        self.fc21 = nn.Linear(400, 20)  # 均值
        self.fc22 = nn.Linear(400, 20)  # 方差

    def forward(self, x):
        x = x.view(-1, input_dim)
        x = torch.relu(self.fc1(x))
        mu = self.fc21(x)
        logvar = self.fc22(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, input_dim)

    def forward(self, z):
        z = torch.relu(self.fc3(z))
        x_recon = torch.sigmoid(self.fc4(z))
        return x_recon


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.fc5 = nn.Linear(20, 400)
        self.fc6 = nn.Linear(400, input_dim)

    def forward(self, z):
        z = torch.relu(self.fc5(z))
        x_gen = torch.sigmoid(self.fc6(z))
        return x_gen

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.fc7 = nn.Linear(input_dim, 400)
        self.fc8 = nn.Linear(400, 1)

    def forward(self, x):
        x = x.view(-1, input_dim)
        x = torch.relu(self.fc7(x))
        prob_real_or_fake = torch.sigmoid(self.fc8(x))
        return prob_real_or_fake

# 定义AVAE模型
class AVAE(nn.Module):
    def __init__(self, encoder, decoder, generator, discriminator):
        super(AVAE, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
        self.discriminator = discriminator

    def forward(self, x):
        # 编码
        mu, logvar = self.encoder(x)

        # 从潜在空间采样
        std = torch.exp(0.5 * logvar)
        epsilon = torch.randn_like(std)
        z = mu + epsilon * std

        # 解码
        x_recon = self.decoder(z)

        return x_recon, mu, logvar

def training(random_state, num_epochs, input_dim, learning_rate, all_omics_tr,
             all_omics_te, avae_model, masked_position, masked_values):

    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    avae_model = AVAE(encoder, decoder, generator, discriminator).to(device)

    num_rows_tr = all_omics_tr.shape[0]
    num_cols_tr = all_omics_tr.shape[1]

    num_rows_te = all_omics_te.shape[0]
    num_cols_te = all_omics_te.shape[1]

    w_row = torch.rand(num_rows_tr, requires_grad=True)
    w_col = torch.rand(num_cols_tr, requires_grad=True)

    optimizer = torch.optim.Adam(avae_model.parameters(), lr=learning_rate)
    optimizer_imp = torch.optim.SGD([w_row, w_col], lr=learning_rate)

    all_omics_tr = np.array(all_omics_tr)
    dataset_tr = dataset(all_omics_tr)
    dataloader_tr = Dataloader.DataLoader(dataset_tr, batch_size=245, shuffle=True)

    all_omics_te = np.array(all_omics_te)
    dataset_te = dataset(all_omics_te)
    dataloader_te = Dataloader.DataLoader(dataset_te, batch_size=106, shuffle=True)

    BCE = nn.BCELoss(reduction='sum')
    MSE = nn.MSELoss(reduction='sum')

    reconstruction_losses = []
    reconstruction_losses_te = []

    # classfier = RandomForestClassifier(n_estimators=RF_tree_num, random_state=0, n_jobs=RF_job)
    # classfier1 = RandomForestClassifier(n_estimators=RF_tree_num, random_state=0, n_jobs=RF_job)

    init_acc = 0
    init_f1 = 0
    init_auc = 0

    # recon_batch_list = []

    for epoch in range(num_epochs):
        recon_batch_list = []

        for batch_idx, DATA in enumerate(dataloader_tr):
            data_tr = DATA
            data_tr = data_tr.to(device)
            data_tr = data_tr.view(-1, input_dim)

            predicted_matrix = torch.zeros_like(data_tr)
            for i in range(num_rows_tr):
                for j in range(num_cols_tr):
                    # if matrix_tensor[i, j] is None:
                    # if data_tr[i, j] == 0:
                    if np.isnan(data_tr[i][j].cpu().detach().numpy()) == True:
                        data_tr_np = data_tr.cpu().detach().numpy()
                        # row_values = data_tr[i][data_tr[i].nonzero(as_tuple=True)]
                        # col_values = data_tr[:, j][data_tr[:, j].nonzero(as_tuple=True)]
                        row_values = data_tr_np[i][~np.isnan(data_tr_np[i])]
                        col_values = data_tr_np[:, j][~np.isnan(data_tr_np[:, j])]
                        row_values, col_values = torch.tensor(row_values), torch.tensor(col_values)
                        predicted_value = (torch.sum(row_values) * w_row[i] + torch.sum(col_values) * w_col[j]) / (
                                len(row_values) + len(col_values))
                        predicted_matrix[i, j] = predicted_value

            data_tr = torch.where(torch.isnan(data_tr), torch.full_like(data_tr, 0), data_tr)
            data_tr = data_tr + predicted_matrix
            data_tr = data_tr.float()

            recon_batch, mu, logvar = avae_model(data_tr)

            recon_loss = MSE(recon_batch, data_tr.view(-1, input_dim))
            kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
            vae_loss = recon_loss + kl_loss

            z_fake = torch.randn(data_tr.shape[0], 20).to(device)
            x_fake = generator(z_fake)
            prob_real = discriminator(data_tr.view(-1, input_dim))
            prob_fake = discriminator(x_fake)
            gan_loss_gen = BCE(prob_fake, torch.ones_like(prob_fake)).to(device)
            gan_loss_disc = BCE(prob_real, torch.ones_like(prob_real)).to(device) +\
                            BCE(prob_fake, torch.zeros_like(prob_fake)).to(device)

            total_loss = vae_loss + gan_loss_gen + gan_loss_disc
            reconstruction_losses.append(recon_loss.item())

            recon_batch_list.append(recon_batch)

            # z = avae_model.reparameterize(mu, logvar)
            # generated_data = avae_model.decode(z)
            # d_real = avae_model.discriminator(data_tr)
            # d_fake = avae_model.discriminator(generated_data.detach())

            # labels_tr_loss = labels_tr.to(device)
            # reconstruction0 = recon_batch.cpu().detach().numpy()
            # labels_tr = labels_tr.cpu().detach().numpy()
            # classfier1.fit(reconstruction0, np.ravel(labels_tr))


            # print("1-loss: ", loss)
            total_loss.requires_grad_(True)
            # loss = avae_loss_function(recon_batch, data_tr, mu, logvar, d_real, d_fake).to(device)
            # loss = avae_loss_function(y_hat_loss, labels_tr_loss, mu, logvar, d_real, d_fake)
            optimizer.zero_grad()
            optimizer_imp.zero_grad()
            total_loss.backward()
            optimizer.step()
            optimizer_imp.step()

            # y_hat = classfier.predict(z)

            # acc = accuracy_score(y_true=labels_tr, y_pred=y_hat)

            # if batch_idx % log_interval == 0:
            #     print('Epoch_training [{}/{}], Batch [{}/{}], Loss: {:.4f}'.format(
            #         epoch + 1, num_epochs, batch_idx + 1, len(dataloader_tr), total_loss.item()))

        # writer.add_scalar("train_loss", total_loss.item(), epoch + 1)
        # writer.add_scalar("train_acc", 1 - total_loss, epoch + 1)

        avae_model.eval()
        with torch.no_grad():

            for batch_idx, DATA_te in enumerate(dataloader_te):
                data_te = DATA_te
                data_te = data_te.to(device)
                data_te = data_te.view(-1, input_dim)

                predicted_matrix = torch.zeros_like(data_te)
                for i in range(num_rows_te):
                    for j in range(num_cols_te):
                        # if matrix_tensor[i, j] is None:
                        # if data_te[i, j] == 0:
                        if np.isnan(data_te[i][j].cpu().detach().numpy()) == True:
                            data_te_np = data_te.cpu().detach().numpy()
                            row_values = data_te_np[i][~np.isnan(data_te_np[i])]
                            col_values = data_te_np[:, j][~np.isnan(data_te_np[:, j])]
                            row_values, col_values = torch.tensor(row_values), torch.tensor(col_values)
                            # row_values = data_te[i][data_te[i].nonzero(as_tuple=True)]
                            # col_values = data_te[:, j][data_te[:, j].nonzero(as_tuple=True)]
                            predicted_value = (torch.sum(row_values) * w_row[i] + torch.sum(col_values) * w_col[j]) / (
                                    len(row_values) + len(col_values))
                            predicted_matrix[i, j] = predicted_value

                data_te = torch.where(torch.isnan(data_te), torch.full_like(data_te, 0), data_te)
                data_te = data_te + predicted_matrix
                data_te = data_te.float()

                recon_batch_te, mu_te, logvar_te = avae_model(data_te)

                recon_loss_te = MSE(recon_batch_te, data_te.view(-1, input_dim))
                kl_loss_te = -0.5 * torch.sum(1 + logvar_te - mu_te.pow(2) - logvar_te.exp())
                vae_loss_te = recon_loss_te + kl_loss_te

                z_fake_te = torch.randn(data_te.shape[0], 20).to(device)
                x_fake_te = generator(z_fake_te)
                prob_real_te = discriminator(data_te.view(-1, input_dim))
                prob_fake_te = discriminator(x_fake_te)
                gan_loss_gen_te = BCE(prob_fake_te, torch.ones_like(prob_fake_te)).to(device)
                gan_loss_disc_te = BCE(prob_real_te, torch.ones_like(prob_real_te)).to(device) + \
                                BCE(prob_fake_te, torch.zeros_like(prob_fake_te)).to(device)

                total_loss_te = vae_loss_te + gan_loss_gen_te + gan_loss_disc_te
                reconstruction_losses_te.append(recon_loss_te.item())

                recon_batch_list.append(recon_batch_te)

                # reconstruction1 = recon_batch_te.cpu().detach().numpy()
                recon_all = torch.concat(recon_batch_list)
                imputed_value = np.array([recon_all[i].item() for i in masked_position])
                cos_sim = imputed_value.dot(masked_values) / (np.linalg.norm(imputed_value) * np.linalg.norm(masked_values))
                rmse = sqrt(mean_squared_error(imputed_value, masked_values))
                pcc = PCC(imputed_value, masked_values)

                print('Epoch_training [{}/{}], Loss_tr:{:.2f}=vae_loss({:.2f})+gen_loss({:.2f})+disc_loss({:.2f}), '
                      'Loss_te:{:.2f}=vae_loss_te({:.2f})+gen_loss_te({:.2f})+disc_loss_te({:.2f}), cos_sim:{:.2f}, RMSE:{:.2f}, PCC:{:.2f}'
                      .format(epoch + 1, num_epochs, total_loss.item(), vae_loss.item(), gan_loss_gen.item(), gan_loss_disc.item(), total_loss_te.item(),
                       vae_loss_te.item(), gan_loss_gen_te.item(), gan_loss_disc_te.item(), cos_sim, rmse, pcc))


                # data_all = torch.concat([data_tr,data_te])
            # recon_all = torch.concat(recon_batch_list)
            recon_all_best = torch.concat(recon_batch_list)

    return recon_all_best

def DSAR(omics_dir, output_dir, random_state, num_epochs, input_dim, learning_rate, test_size, shuffle_dataset, mask_rate, mask_method='MCAR'):

    setup_seed(random_state)

    # all_omics, mRNA, meth, miRNA, labels = input_data_R(mRNA_dir, meth_dir, miRNA_dir, labels_dir)
    all_omics = omics_load(omics_dir)
    all_omics = pd.DataFrame(all_omics)

    if mask_method == 'MCAR':
        maskded_matrix, masked_values, masked_position = MCAR(mask_rate, all_omics)
    elif mask_method == 'MNAR':
        maskded_matrix, masked_values, masked_position = MNAR(mask_rate, all_omics)

    all_omics_tr, all_omics_te = train_test_split(maskded_matrix, test_size=test_size, shuffle=shuffle_dataset, random_state=random_state)

    encoder = Encoder().to(device)
    decoder = Decoder().to(device)
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    avae_model = AVAE(encoder, decoder, generator, discriminator).to(device)

    matrix = training(random_state, num_epochs, input_dim, learning_rate, all_omics_tr,
                            all_omics_te, avae_model, masked_position, masked_values)

    matrix_np = matrix.cpu().detach().numpy()
    matrix_pd = pd.DataFrame(matrix_np)
    matrix_pd.to_csv(output_dir)

    return matrix_pd


if __name__ == "__main__":
    omics_dir = [r"E:\pytorch\XRN\dataset\ROSMAP_data\ROSMAP_1.csv", r"E:\pytorch\XRN\dataset\ROSMAP_data\ROSMAP_2.csv", r"E:\pytorch\XRN\dataset\ROSMAP_data\ROSMAP_3.csv"]
    # omics_dir = [r"E:\pytorch\XRN\dataset\ROSMAP_data1\ROSMAP_3.csv"]
    output_dir = r"E:\pytorch\XRN\dataset\ROSMAP_data\matrx.csv"

    input_dim = omics_load(omics_dir).shape[1]
    learning_rate = 0.0005  # 0.0005
    num_epochs = 100
    random_state = 0
    test_size = 0.3
    shuffle_dataset = False  # True or False
    masked_rate = 0.01  # 0-1
    mask_method = 'MNAR'  # MNAR or MCAR
    # ROSMAP
    # mRNA_dir = r"E:\pytorch\XRN\dataset\ROSMAP_data1\ROSMAP_1.csv"
    # meth_dir = r"E:\pytorch\XRN\dataset\ROSMAP_data1\ROSMAP_2.csv"
    # miRNA_dir = r"E:\pytorch\XRN\dataset\ROSMAP_data1\ROSMAP_3.csv"

    DSAR(omics_dir, output_dir, random_state, num_epochs, input_dim, learning_rate, test_size, shuffle_dataset, masked_rate, mask_method)



