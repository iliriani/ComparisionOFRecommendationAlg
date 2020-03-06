from sklearn.neighbors import NearestNeighbors
import pandas as pd
import numpy as np
import random as rand
from scipy.sparse.linalg import svds

# Lexojme te dhenat nga nje dataset i marr nga MovieLens per vleresimin e filmave
ratings_list = [i.strip().split("::") for i in open('C:\\Users\\admin\PycharmProjects\MasterProject\ml-1m\\ratings.dat', 'r').readlines()]
users_list = [i.strip().split("::") for i in open('C:\\Users\\admin\PycharmProjects\MasterProject\ml-1m\\users.dat', 'r').readlines()]
movies_list = [i.strip().split("::") for i in open('C:\\Users\\admin\PycharmProjects\MasterProject\ml-1m\movies.dat', 'r').readlines()]

# Te gjitha te dhenat e lexuara i kthejme ne dataframe per manipulim me te lehete
ratings_df = pd.DataFrame(ratings_list, columns = ['UserID', 'MovieID', 'Rating', 'Timestamp'], dtype = int)
movies_df = pd.DataFrame(movies_list, columns = ['MovieID', 'Title', 'Genres'])

# Kthehen gjitha te dhenat e movie_df ne koloene MovieID ne vlera numerike
movies_df['MovieID'] = movies_df['MovieID'].apply(pd.to_numeric)

# Te gjitha te dhenat qe per filma gjenden ne rreshta, tani apo kthehen ne kolona dhe ne rreshta jane perdoresit
R_df = ratings_df.pivot(index = 'UserID', columns ='MovieID', values = 'Rating').fillna(0)


useri = 735
tabela_userit = []
# Gjej gjith filmat qe jane vleresuar me 5
# filmat_userit = R_df.loc[useri,:]>4
vlersimet_userit = R_df.loc[useri,R_df.loc[useri,:]>4]

for i in range(vlersimet_userit.shape[0]):
    gjitha = useri, vlersimet_userit.index[i], vlersimet_userit.iloc[i]
    tabela_userit.append(gjitha)

print(tabela_userit)

# E zgjedhim rastesisht nje film te vleresuar me 5 e mbishkruajme me 0
numri_rastesishem = rand.randint(0,len(tabela_userit)-1)
# R_tjeter = R_df.copy()
filmi_vleresuar = tabela_userit[numri_rastesishem][1]
# filmi_vleresuar = 858
R_df.at[useri,filmi_vleresuar] = 0

ratings_df.at[(ratings_df.UserID == useri) & (ratings_df.MovieID == filmi_vleresuar)] = 0


# E bejme normalizimin e te dhenave duke e zbritur mesatren e rreshtit per secilin rresht
R = R_df.as_matrix()
user_ratings_mean = np.mean(R, axis = 1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)

R_final = pd.DataFrame(R_demeaned)

# Per SVD algoritmin
U, sigma, Vt = svds(R_demeaned, k=50)
sigma = np.diag(sigma)

# Ia shktojme prap vleren mesatare rrshtave ashtu qe ta kemi vleresimin e sakte 1-5
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
preds_df = pd.DataFrame(all_user_predicted_ratings, columns=R_df.columns)

knn_afersia_5 = []
# knn_afersia_2 = []

svd_afersia_5 = []
# svd_afersia_2 = []

mesazhi = ''
ska_gjetje_knn = False
ska_gjetje_svd = False
def knn_rekomandimi(M):

    model_knn_2 = NearestNeighbors(metric = 'cosine', algorithm = 'brute')
    model_knn_2.fit(M)

    distances, indices = model_knn_2.kneighbors(M.iloc[useri,:].values.reshape(1,-1), n_neighbors = 3952)

    row,col = M.shape
    # print(R_final_2.head(10))

    for i in range(0, len(distances.flatten())):
        if i == 0:
            print('Rekomadimet ne KNN per {0}:\n'.format(R_df.index[useri]))
        else:
            indeksi =  M.index[indices.flatten()[i]]
            # indeksi = indices.flatten()[i]
            kol = col if (indeksi% col) == 0 else (indeksi% col)
            filmi = movies_df.iloc[kol]['Title']
            if kol == filmi_vleresuar:
                afersia_5 = i, distances.flatten()[i],kol
                knn_afersia_5.append(afersia_5)
                print(knn_afersia_5)


       

def svd_rekomandimi(predictions_df, userID, movies_df, original_ratings_df, num_recommendations=3952):

    # Merr dhe rendite rekomadimet per userin
    user_row_number = userID - 1  # Id e Userit fillon nga 1
    sorted_user_predictions = predictions_df.iloc[user_row_number].sort_values(ascending=False)

    # Get the user's data and merge in the movie information.
    user_data = original_ratings_df[original_ratings_df.UserID == (userID)]
    user_full = (user_data.merge(movies_df, how='left', left_on='MovieID', right_on='MovieID').
                    sort_values(['Rating'], ascending=False)
                    )

    print("Perdoruesi {0} ka vleresuar deri me tani {1} filma.".format(userID, user_full.shape[0]))
    print("Rekomandohen {0} filma qe perdoresi akoma nuk i ka pare.".format(num_recommendations))

    # Jepen rekomandimet me vlerat me te larta qe perdoruesi akoma nuk i ka pare
    recommendations = (movies_df[~movies_df['MovieID'].isin(user_full['MovieID'])].
                        merge(pd.DataFrame(sorted_user_predictions).reset_index(), how='left',
                                left_on='MovieID',
                                right_on='MovieID').
                        rename(columns={user_row_number: 'Predictions'}).
                        sort_values('Predictions', ascending=False).
                        iloc[:num_recommendations, :-1]
                        )

    # rekomandimet_df = pd.DataFrame(recommendations)
    # rekomandimet_df.set_index('MovieID')
    r,k = recommendations.shape

    for i in range(r):

        if recommendations.iloc[i,0] == filmi_vleresuar:
        # print(recommendations.iloc[i,1])
            afersia_5_svd = i,filmi_vleresuar,recommendations.iloc[i,1]
            svd_afersia_5.append(afersia_5_svd)
            print(svd_afersia_5)
        # else:
        #     err_svd_mesazhi = 'SVD nuk arriti me e gjet ne 100 iterimet e para rekomadnimin'
        #     print(err_svd_mesazhi)
    return user_full, recommendations



already_rated, predictions = svd_rekomandimi(preds_df, useri, movies_df, ratings_df,3952)

knn_preds = knn_rekomandimi(R_final)

if len(knn_afersia_5) == 0:
    print('knn ska gjete sen per ',filmi_vleresuar)
if len(svd_afersia_5) == 0:
    print('svd ska gjet sen per ',filmi_vleresuar)
