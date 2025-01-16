from modelscope.hub.constants import Licenses, ModelVisibility
from modelscope.hub.api import HubApi

YOUR_ACCESS_TOKEN = "c1e804ea-36b5-4e62-b20a-672b48361817"
api = HubApi()
api.login(YOUR_ACCESS_TOKEN)

owner_name = 'ubowang'
model_name = 'MAmmoTH-Critique-1'
model_id = f"{owner_name}/{model_name}"

api.create_model(
    model_id,
    visibility=ModelVisibility.PUBLIC,
    license=Licenses.APACHE_V2,
    chinese_name="测试"
)