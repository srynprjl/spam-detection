from dash import Dash, html, dcc, Input, Output, State
from collections import Counter
import pickle

def get_gaussian_log_prob(x, mean, var):
    coeff = -0.5 * np.log(2 * np.pi * var)
    exponent = -0.5 * ((x - mean)**2 / var)
    return coeff + exponent

def test(message, m):
    chars = len(message)
    words_list = message.split()
    words_count = len(words_list) if len(words_list) > 0 else 1
    feat = {
        'digit_ratio': sum(c.isdigit() for c in message) / chars if chars > 0 else 0,
        'has_url': 1 if any(x in message.lower() for x in ['http', 'www', '.com']) else 0,
        'lexical_diversity': len(set(words_list)) / words_count,
        'avg_word_len': chars / words_count
    }
    counts = Counter(words_list)
    vec = np.array([counts[word] for word in m["vocab"]])

    t_score_spam = m["log_prior_spam"] + (vec @ m["log_prob_spam"])
    t_score_ham = m["log_prior_ham"] + (vec @ m["log_prob_ham"])

    e_score_spam = sum(get_gaussian_log_prob(feat[col], m['spam_stats']['mean'][col], m['spam_stats']['var'][col]) for col in m["eng_cols"])
    e_score_ham = sum(get_gaussian_log_prob(feat[col], m['ham_stats']['mean'][col], m['ham_stats']['var'][col]) for col in m["eng_cols"])
    is_spam = (t_score_spam + e_score_spam) > (t_score_ham + e_score_ham)
    return ("SPAM", 1) if is_spam else ("HAM", 0)

with open("./models/spam_detection.pkl", "rb") as f:
  m = pickle.load(f)

app = Dash(__name__)

app.layout = html.Div(
    style={
        "minHeight": "100vh",
        "backgroundColor": "#1f2937",
        "color": "white",
        "display": "flex",
        "flexDirection": "column",
        "alignItems": "center",
        "justifyContent": "center",
        "padding": "20px"
    },
    children=[

        html.H1(
            "Spam Detection System",
            style={
                "fontSize": "40px",
                "fontWeight": "bold",
                "marginBottom": "30px",
                "textAlign": "center"
            }
        ),

        html.Div(
            style={
                "maxWidth": "400px",
                "width": "100%",
                "display": "flex",
                "flexDirection": "column",
                "gap": "20px",
                "backgroundColor": "#374151",
                "padding": "25px",
                "borderRadius": "10px",
                "boxShadow": "0px 4px 10px rgba(0,0,0,0.5)"
            },
            children=[
                html.Div([
                    html.Label("Message:", style={"fontSize": "18px"}),
                    dcc.Textarea(
                        id="message-input",
                        placeholder="Enter a Message",
                        style={
                            "width": "100%",
                            "height": "120px",
                            "marginTop": "8px",
                            "padding": "10px",
                            "borderRadius": "6px",
                            "border": "none",
                            "color": "black"
                        }
                    )
                ]),

                html.Div(
                    style={"textAlign": "center"},
                    children=[
                        html.Button(
                            "Submit",
                            id="submit-button",
                            n_clicks=0,
                            style={
                                "fontSize": "18px",
                                "backgroundColor": "#1f2937",
                                "color": "white",
                                "padding": "12px 20px",
                                "borderRadius": "6px",
                                "border": "none",
                                "cursor": "pointer"
                            }
                        )
                    ]
                ),

                html.Div(
                    id="output-result",
                    style={
                        "marginTop": "10px",
                        "fontSize": "20px",
                        "fontWeight": "bold",
                        "textAlign": "center"
                    }
                )
            ]
        )
    ]
)

@app.callback(
    Output("output-result", "children"),
    Input("submit-button", "n_clicks"),
    State("message-input", "value")
)
def show_output(n_clicks, message):
    if n_clicks == 0 or not message:
        return ""
    return f"Result: {test(message, m)}"


if __name__ == "__main__":
    app.run(debug=True)