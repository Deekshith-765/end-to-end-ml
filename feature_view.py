from feast import Entity, FeatureView, Field
from feast.types import Int64, Float32
from feast.data_source import FileSource

user = Entity(name="user_id", join_keys=["user_id"])

# Data source
data_source = FileSource(
    path="data/train.csv",
    timestamp_field="event_timestamp"
)

user_features = FeatureView(
    name="user_features",
    entities=[user],
    schema=[
        Field(name="age", dtype=Int64),
        Field(name="salary", dtype=Float32),
    ],
    source=data_source,
)