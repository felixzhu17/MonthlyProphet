MONTHS = {
    "January": 1,
    "February": 2,
    "March": 3,
    "April": 4,
    "May": 5,
    "June": 6,
    "July": 7,
    "August": 8,
    "September": 9,
    "October": 10,
    "November": 11,
    "December": 12,
}
import warnings

import pandas as pd
import plotly.express as px
import plotly.graph_objs as go

try:
    from prophet import Prophet
except:
    warnings.warn("Need to install prophet for uploading plots")


class MonthlyProphetForecast:
    def __init__(
        self,
        df,
        column,
        forecast_periods=12,
        cap=None,
        seasonality_mode="multiplicative",
        plot_confidence_interval=False,
        months=None,
        interval_width=0.8,
        backtest=None,
        *args,
        **kwargs
    ):
        self.df = df
        self.column = column
        self.cap = cap
        self.forecast_periods = forecast_periods
        self.seasonality_mode = seasonality_mode
        self.plot_confidence_interval = plot_confidence_interval
        self.months = (
            MONTHS
            if months is None
            else {k: v for k, v in MONTHS.items() if k in months}
        )
        self.interval_width = interval_width
        self.backtest = backtest
        self.interval_label = self._format_percentage((1 - interval_width) / 2)
        # self._excluded_months = [
        #     MONTHS[i] for i in MONTHS.keys() if i not in self.months.keys()
        # ]
        self.model_df = self.prepare_df(self.df, self.column, self.cap)
        self.sim_start = self.model_df.index.max()
        self.m = self.fit_prophet_model(self.model_df, self.cap, *args, **kwargs)

        if self.cap:
            self.cap_df = self.prepare_df(self.df, self.cap, None)
            self.cap_m = self.fit_prophet_model(self.cap_df, None, *args, **kwargs)
            self.cap_forecast = self.get_forecast(self.cap_m, None)

        self.forecast = self.get_forecast(
            self.m, self.cap_forecast if self.cap else None
        )
        self.plot_df = self.get_plot_df()
        self.seasonal_df = self.get_seasonal_df()
        self.historical_plot = self.get_historical_plot()
        self.forecast_plot = self.get_forecast_plot()
        self.historical_trend_plot = self.get_historical_trend_plot()
        self.trend_plot = self.get_trend_plot()
        self.seasonal_plot = self.get_seasonal_plot()

    def prepare_df(self, df, column, cap):
        model_df = df.copy()
        if self.backtest is not None:
            model_df = model_df.loc[: self.backtest]
        model_df = model_df[[column]]
        model_df["ds"] = model_df.index
        model_df.columns = ["y", "ds"]
        if cap:
            model_df["cap"] = self.df[cap]
        model_df = self._append_month_cols(model_df)
        return model_df

    def fit_prophet_model(self, df, cap, *args, **kwargs):
        if cap:
            kwargs["growth"] = "logistic"
        m = Prophet(
            yearly_seasonality=False,
            seasonality_mode=self.seasonality_mode,
            interval_width=self.interval_width,
            *args,
            **kwargs
        )
        [m.add_regressor(month) for month in self.months.keys()]
        # if len(self._excluded_months) > 0:
        #     m.add_regressor("Overall")
        m.fit(df)
        return m

    def get_forecast(self, m, cap):
        future = m.make_future_dataframe(
            periods=self.forecast_periods, freq="M", include_history=True
        )
        future = self._append_month_cols(future)
        if cap is not None:
            future["cap"] = list(self.cap_forecast.yhat)
        forecast = m.predict(future)
        forecast.index = forecast.ds
        return forecast

    def get_plot_df(self):
        temp = self.forecast[["yhat", "yhat_lower", "yhat_upper"]]
        return pd.concat([self.df, temp])

    def get_seasonal_df(self):
        seasonality = [
            {"Month": month, "Value": self._get_seasonal_value(self.forecast[month])}
            for month in self.months
        ]
        # if len(self._excluded_months) > 0:
        #     seasonality.append(
        #         {
        #             "Month": "Overall",
        #             "Value": self._get_seasonal_value(self.forecast["Overall"]),
        #         }
        #     )
        seasonal_df = pd.DataFrame(seasonality)
        if self.seasonality_mode != "multiplicative":
            self.seasonal_adjust = seasonal_df["Value"].mean()
            self.forecast["trend"] = self.forecast["trend"] + self.seasonal_adjust
            self.forecast["trend_lower"] = (
                self.forecast["trend_lower"] + self.seasonal_adjust
            )
            self.forecast["trend_upper"] = (
                self.forecast["trend_upper"] + self.seasonal_adjust
            )
            seasonal_df["Value"] = seasonal_df["Value"] - self.seasonal_adjust
        else:
            seasonal_df["Value"] = seasonal_df["Value"] + 1
            self.seasonal_adjust = seasonal_df["Value"].mean()
            self.forecast["trend"] = self.forecast["trend"] * self.seasonal_adjust
            self.forecast["trend_lower"] = (
                self.forecast["trend_lower"] * self.seasonal_adjust
            )
            self.forecast["trend_upper"] = (
                self.forecast["trend_upper"] * self.seasonal_adjust
            )
            seasonal_df["Value"] = seasonal_df["Value"] / self.seasonal_adjust
        return seasonal_df

    def get_historical_plot(self):
        fig = go.Figure()
        fig.add_scattergl(
            x=self.plot_df.index.where(self.plot_df.index <= self.sim_start),
            y=self.plot_df[self.column].where(self.plot_df.index <= self.sim_start),
            line={"color": "#091E42"},
            name="Actual",
        )
        fig.update_layout(
            xaxis_title="Date",
            plot_bgcolor="white",
        )
        return fig

    def get_forecast_plot(self):
        fig = go.Figure()
        fig.add_scattergl(
            x=self.plot_df.index,
            y=self.plot_df[self.column],
            line={"color": "#091E42"},
            name="Actual",
        )
        fig.add_scattergl(
            x=self.plot_df.index,
            y=self.plot_df.yhat.where(self.plot_df.index <= self.sim_start),
            line={"color": "#FF5630"},
            name="Predicted",
        )
        fig.add_scattergl(
            x=self.plot_df.index,
            y=self.plot_df.yhat.where(self.plot_df.index > self.sim_start),
            line={"color": "#0052CC"},
            name="Forecast",
        )
        if self.plot_confidence_interval:
            fig.add_scattergl(
                x=self.plot_df.index,
                y=self.plot_df.yhat_lower.where(self.plot_df.index > self.sim_start),
                line={"color": "#C1C7D0", "dash": "dash"},
                name=self.interval_label,
            )
            fig.add_scattergl(
                x=self.plot_df.index,
                y=self.plot_df.yhat_upper.where(self.plot_df.index > self.sim_start),
                line={"color": "#C1C7D0", "dash": "dash"},
                name=self.interval_label,
            )
        fig.update_layout(
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            legend_title_text="",
            xaxis_title="Date",
            plot_bgcolor="white",
        )
        return fig

    def get_historical_trend_plot(self):
        fig = go.Figure()
        fig.add_scattergl(
            x=self.forecast.index.where(self.forecast.index <= self.sim_start),
            y=self.forecast.trend.where(self.forecast.index <= self.sim_start),
            line={"color": "#091E42"},
            name="Trend",
        )
        fig.update_layout(
            xaxis_title="Date",
            plot_bgcolor="white",
        )
        return fig

    def get_trend_plot(self):
        fig = go.Figure()
        fig.add_scattergl(
            x=self.forecast.index,
            y=self.forecast.trend.where(self.forecast.index <= self.sim_start),
            line={"color": "#091E42"},
            name="Trend",
        )
        fig.add_scattergl(
            x=self.forecast.index,
            y=self.forecast.trend.where(self.forecast.index > self.sim_start),
            line={"color": "#0052CC"},
            name="Forecast",
        )
        if self.plot_confidence_interval:
            fig.add_scattergl(
                x=self.forecast.index,
                y=self.forecast.trend_lower.where(self.forecast.index > self.sim_start),
                line={"color": "#C1C7D0", "dash": "dash"},
                name=self.interval_label,
            )
            fig.add_scattergl(
                x=self.forecast.index,
                y=self.forecast.trend_upper.where(self.forecast.index > self.sim_start),
                line={"color": "#C1C7D0", "dash": "dash"},
                name=self.interval_label,
            )

        fig.update_layout(
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
            legend_title_text="",
            xaxis_title="Date",
            plot_bgcolor="white",
        )
        return fig

    def get_seasonal_plot(self):
        fig = px.line(self.seasonal_df, x="Month", y="Value")
        fig.update_layout(plot_bgcolor="white")
        fig.update_xaxes(title_text="Month")
        fig.update_yaxes(title_text="Seasonal Trend")
        return fig

    def _append_month_cols(self, df):
        for month in self.months:
            df[month] = (df["ds"].dt.month == self.months[month]).values.astype("float")
        # if len(self._excluded_months) > 0:
        #     df["Overall"] = (
        #         df["ds"].dt.month.isin(self._excluded_months)
        #     ).values.astype("float")
        return df

    def _get_seasonal_value(self, col):
        values = [i for i in col.unique() if i != 0]
        if len(values) != 1:
            raise ValueError("More than one unique value for season")
        return values[0]

    def _format_percentage(self, num):
        return "{:.0%}".format(num)
