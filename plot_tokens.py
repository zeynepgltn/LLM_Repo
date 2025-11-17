import plotly.graph_objects as go
import plotly.offline

def plot_tokens(sentences_data, title, dims=[0, 1, 2]):
  data = [
    go.Scatter3d(
      x=sentence_data["words"][:, dims[0]],
      y=sentence_data["words"][:, dims[1]],
      z=sentence_data["words"][:, dims[2]],
      mode="markers+text",
      marker=dict(
        size=6,
        color=sentence_data["color"],
      ),
      text=sentence_data["labels"],
      hoverinfo="text",
    ) for sentence_data in sentences_data
  ]

  layout = go.Layout(
    scene=dict(
      xaxis_title="Sertlik",
      yaxis_title="Parlaklık",
      zaxis_title="Kırmızılık",
    ),
    title=title,
  )

  fig = go.Figure(data=data, layout=layout)
  plotly.offline.iplot(fig)