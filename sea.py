import seaborn as sns
import matplotlib.pyplot as plt


tips = sns.load_dataset("tips")
sns.scatterplot(x="total_bill", y="tip", data=tips)
plt.show()


tips = sns.load_dataset("tips")
sns.lineplot(x="total_bill", y="tip", data=tips)
plt.show()


tips = sns.load_dataset("tips")
sns.boxplot(x="day", y="total_bill", data=tips)
plt.show()

tips = sns.load_dataset("tips")
sns.violinplot(x="day", y="total_bill", data=tips)
plt.show()


tips = sns.load_dataset("tips")
sns.histplot(tips["total_bill"], bins=30, kde=True)
plt.show()

tips = sns.load_dataset("tips")
sns.kdeplot(tips["total_bill"])
plt.show()

tips = sns.load_dataset("tips")
sns.pairplot(tips)
plt.show()



tips = sns.load_dataset("tips")
sns.stripplot(x="day", y="total_bill", data=tips, jitter=True, marker="o", color="skyblue")
plt.title("Strip Plot of Total Bill by Day")
plt.show()



tips = sns.load_dataset("tips")
sns.barplot(x="day", y="total_bill", data=tips)
plt.show()

tips = sns.load_dataset("tips")
sns.countplot(x="day", data=tips)
plt.show()

tips = sns.load_dataset("tips")
sns.lmplot(x="total_bill", y="tip", data=tips)
plt.show()

tips = sns.load_dataset("tips")
sns.regplot(x="total_bill", y="tip", data=tips)
plt.show()

tips = sns.load_dataset("tips")
sns.jointplot(x="total_bill", y="tip", data=tips)
plt.show()


tips = sns.load_dataset("tips")
sns.catplot(x="day", y="total_bill", kind="box", data=tips)
plt.show()

tips = sns.load_dataset("tips")
sns.lmplot(x="total_bill", y="tip", hue="day", data=tips)
plt.show()