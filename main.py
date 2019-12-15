import pandas as pd
import sklearn.preprocessing as pp
import sklearn.linear_model as lm
import sklearn.metrics as sm
import sklearn.model_selection as ms

muscle_mass_df = pd.read_csv("data/muscle_mass.csv")
muscle_mass_df.sort_values(by="training_time", inplace=True)
print(muscle_mass_df.head())

class OptimalRegressionFinder:
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.output = {}

    def get_MSE(self, model, X, y_true):
        y_predicted = model.predict(X)
        MSE = sm.mean_squared_error(y_true, y_predicted)
        return MSE

    def find_best_degree(self, max_degree):
        X_train, X_test, y_train, y_test = ms.train_test_split(self.X, self.y, shuffle=True)

        for z in range(1, 100):
            polynomial_transformer = pp.PolynomialFeatures(degree=z)
            X_transformed_train = polynomial_transformer.fit_transform(X_train)
            X_transformed_test = polynomial_transformer.fit_transform(X_test)

            muscle_mass_model = lm.LinearRegression()
            muscle_mass_model.fit(X_transformed_train, y_train)

            # print(muscle_mass_model.coef_)
            # print(muscle_mass_model.intercept_)
            self.output[z] = self.get_MSE(muscle_mass_model, X_transformed_test, y_test)
            print("Train MSE for {} = {}".format(z, self.output[z]))

    def print_best_result(self):
        print(min(self.output.items(), key=lambda x: x[1]))


finder = OptimalRegressionFinder(muscle_mass_df[["training_time"]], muscle_mass_df[["muscle_mass"]])
finder.find_best_degree(10)
finder.print_best_result()