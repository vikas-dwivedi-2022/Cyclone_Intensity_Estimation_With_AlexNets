import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import mean_absolute_error

#------------------------------------------------------#
#           Global AlexNet Predictions                 #
#------------------------------------------------------#

Glo_df1 = pd.read_csv("Predictions_Test_11.csv")
Glo_df2 = pd.read_csv("Predictions_Test_12.csv")
Glo_df3 = pd.read_csv("Predictions_Test_13.csv")
Glo_df4 = pd.read_csv("Predictions_Test_14.csv")
Glo_df5 = pd.read_csv("Predictions_Test_15.csv")
Glo_df6 = pd.read_csv("Predictions_Test_16.csv")
Glo_df7 = pd.read_csv("Predictions_Test_17.csv")
Glo_df8 = pd.read_csv("Predictions_Test_18.csv")
Glo_df9 = pd.read_csv("Predictions_Test_19.csv")
Glo_df10 = pd.read_csv("Predictions_Test_20.csv")

Glo_df1['Glo_Avg'] = (Glo_df1['Pred']+Glo_df2['Pred']+Glo_df3['Pred']+Glo_df4['Pred']+Glo_df5['Pred']+Glo_df6['Pred']+Glo_df7['Pred']+Glo_df8['Pred']+Glo_df9['Pred']+Glo_df10['Pred']) * 0.1

Glo_df1 = Glo_df1.drop(columns=['Pred'])



#------------------------------------------------------#
#                     TAGGING                          #
#------------------------------------------------------#
# Define the categories and their corresponding ranges
categories = {
    1: (10, 33),
    2: (34, 63),
    3: (64, 82),
    4: (83, 95),
    5: (96, 200)
}

#------------------#
# True Tags        #
#------------------#
Glo_df1['True_Cat'] = pd.cut(Glo_df1['True'], bins=[10,33,63,82,95,200], labels=False) + 1
#------------------#
# Predicted Tags   #
#------------------#
Glo_df1['Pred_Cat'] = pd.cut(Glo_df1['Glo_Avg'], bins=[10,33,63,82,95,200], labels=False) + 1

exp_df1 = pd.read_csv("Pred_expB_01.csv")
exp_df2 = pd.read_csv("Pred_expB_02.csv")
exp_df3 = pd.read_csv("Pred_expB_03.csv")
exp_df4 = pd.read_csv("Pred_expB_04.csv")
exp_df5 = pd.read_csv("Pred_expB_05.csv")


exp_df1.rename(columns={'Pred': 'Pred_1'}, inplace=True)
exp_df2.rename(columns={'Pred': 'Pred_2'}, inplace=True)
exp_df3.rename(columns={'Pred': 'Pred_3'}, inplace=True)
exp_df4.rename(columns={'Pred': 'Pred_4'}, inplace=True)
exp_df5.rename(columns={'Pred': 'Pred_5'}, inplace=True)


combined_df = pd.merge(Glo_df1, exp_df1[['Image ID', 'Pred_1']], on='Image ID', how='left')
combined_df = pd.merge(combined_df, exp_df2[['Image ID', 'Pred_2']], on='Image ID', how='left')
combined_df = pd.merge(combined_df, exp_df3[['Image ID', 'Pred_3']], on='Image ID', how='left')
combined_df = pd.merge(combined_df, exp_df4[['Image ID', 'Pred_4']], on='Image ID', how='left')
combined_df = pd.merge(combined_df, exp_df5[['Image ID', 'Pred_5']], on='Image ID', how='left')

#------------------------------------#
# One Hot Encoding (Gating Function) #
#------------------------------------#

one_hot_encoded = pd.get_dummies(combined_df['Pred_Cat'], prefix='Pred_Cat')
combined_df = pd.concat([combined_df, one_hot_encoded], axis=1)


combined_df['MoA'] = combined_df['Pred_Cat_1'] * combined_df['Pred_1'] + combined_df['Pred_Cat_2'] * combined_df['Pred_2']+ combined_df['Pred_Cat_3'] * combined_df['Pred_3']+combined_df['Pred_Cat_4'] * combined_df['Pred_4']+combined_df['Pred_Cat_5'] * combined_df['Pred_5']


combined_df['Both'] = 0.5*(combined_df['MoA']+combined_df['Glo_Avg'])



#------------------------------------------------

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(combined_df['True'], combined_df['Both']))
print(f'Root Mean Square Error (RMSE): {rmse}')

rrmse=rmse/np.sqrt(mean_squared_error(0*combined_df['True'], combined_df['Both']))
print(f'Relative Root Mean Square Error (RRMSE): {rrmse}')

# Calculate mean absolute error
mae = mean_absolute_error(combined_df['True'], combined_df['Both'])
# Print the mean absolute error
print(f'Mean Absolute Error (MAE): {mae}')

# Calculate mean error
mean_error = (combined_df['Both'] - combined_df['True']).mean()

# Print the mean error
print(f'Mean Error: {mean_error}')

# Calculate R-squared score
r2_value = r2_score(combined_df['True'], combined_df['Both'])

# Print the R-squared score
print(f'R2 Score: {r2_value}')
#-------------------------------------------------
plt.figure(figsize=(14, 8))
plt.scatter(combined_df['True'], combined_df['Both'],color='black', alpha=0.3)
plt.plot(combined_df['True'], combined_df['True'], color='red', linestyle='--', label='Y=X Line')  # Straight line y=x
#plt.plot(df1['True'], df1['True']+10, color='black', linestyle='--', label='Y=X+10 Line')  # Straight line y=x+8
#plt.plot(df1['True'], df1['True']-10, color='black', linestyle='--', label='Y=X-10 Line')  # Straight line y=x-8
plt.title('Performance on Test Data', fontsize=25)
plt.xlabel('True Values', fontsize=25)# 45
plt.ylabel('Predicted Values', fontsize=25)
plt.legend() 
#plt.grid()

# Create a hexbin plot on the second subplot
plt.hexbin(combined_df['True'], combined_df['Both'], gridsize=50, cmap='YlGnBu', label='Hexbin Plot',alpha=0.5)
plt.colorbar(label='Point Density')
plt.savefig('final_scatter_plot_both_ovfix.jpg', dpi=300, bbox_inches='tight')
plt.show()


# Calculate RMSE
rmse = np.sqrt(mean_squared_error(combined_df['True'], combined_df['MoA']))
print(f'Root Mean Square Error (RMSE): {rmse}')

rrmse=rmse/np.sqrt(mean_squared_error(0*combined_df['True'], combined_df['MoA']))
print(f'Relative Root Mean Square Error (RRMSE): {rrmse}')

# Calculate mean absolute error
mae = mean_absolute_error(combined_df['True'], combined_df['MoA'])
# Print the mean absolute error
print(f'Mean Absolute Error (MAE): {mae}')

# Calculate mean error
mean_error = (combined_df['MoA'] - combined_df['True']).mean()

# Print the mean error
print(f'Mean Error: {mean_error}')

# Calculate R-squared score
r2_value = r2_score(combined_df['True'], combined_df['MoA'])

# Print the R-squared score
print(f'R2 Score: {r2_value}')



plt.figure(figsize=(14, 8))
plt.scatter(combined_df['True'], combined_df['MoA'],color='black', alpha=0.3)
plt.plot(combined_df['True'], combined_df['True'], color='red', linestyle='--', label='Y=X Line')  # Straight line y=x
#plt.plot(df1['True'], df1['True']+10, color='black', linestyle='--', label='Y=X+10 Line')  # Straight line y=x+8
#plt.plot(df1['True'], df1['True']-10, color='black', linestyle='--', label='Y=X-10 Line')  # Straight line y=x-8
plt.title('Performance on Test Data', fontsize=25)
plt.xlabel('True Values', fontsize=25)# 45
plt.ylabel('Predicted Values', fontsize=25)
plt.legend() 
#plt.grid()

# Create a hexbin plot on the second subplot
plt.hexbin(combined_df['True'], combined_df['MoA'], gridsize=50, cmap='YlGnBu', label='Hexbin Plot',alpha=0.5)
plt.colorbar(label='Point Density')
plt.savefig('final_scatter_plot_MoA_ovfix.jpg', dpi=300, bbox_inches='tight')
plt.show()



# Calculate RMSE
rmse = np.sqrt(mean_squared_error(combined_df['True'], combined_df['Glo_Avg']))
print(f'Root Mean Square Error (RMSE): {rmse}')

rrmse=rmse/np.sqrt(mean_squared_error(0*combined_df['True'], combined_df['Glo_Avg']))
print(f'Relative Root Mean Square Error (RRMSE): {rrmse}')

# Calculate mean absolute error
mae = mean_absolute_error(combined_df['True'], combined_df['Glo_Avg'])
# Print the mean absolute error
print(f'Mean Absolute Error (MAE): {mae}')

# Calculate mean error
mean_error = (combined_df['Glo_Avg'] - combined_df['True']).mean()

# Print the mean error
print(f'Mean Error: {mean_error}')

# Calculate R-squared score
r2_value = r2_score(combined_df['True'], combined_df['Glo_Avg'])

# Print the R-squared score
print(f'R2 Score: {r2_value}')



plt.figure(figsize=(14, 8))
plt.scatter(combined_df['True'], combined_df['Glo_Avg'],color='black', alpha=0.3)
plt.plot(combined_df['True'], combined_df['True'], color='red', linestyle='--', label='Y=X Line')  # Straight line y=x
#plt.plot(df1['True'], df1['True']+10, color='black', linestyle='--', label='Y=X+10 Line')  # Straight line y=x+8
#plt.plot(df1['True'], df1['True']-10, color='black', linestyle='--', label='Y=X-10 Line')  # Straight line y=x-8
plt.title('Performance on Test Data', fontsize=25)
plt.xlabel('True Values', fontsize=25)# 45
plt.ylabel('Predicted Values', fontsize=25)
plt.legend() 
#plt.grid()

# Create a hexbin plot on the second subplot
plt.hexbin(combined_df['True'], combined_df['Glo_Avg'], gridsize=50, cmap='YlGnBu', label='Hexbin Plot',alpha=0.5)
plt.colorbar(label='Point Density')
plt.savefig('final_scatter_plot_ensemble.jpg', dpi=300, bbox_inches='tight')
plt.show()


pred_path = "Result.csv"
combined_df.to_csv(pred_path, index=False)


