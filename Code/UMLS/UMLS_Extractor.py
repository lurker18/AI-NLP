import umls_api
import pandas as pd

resp = umls_api.API(api_key = '2cac3304-aeb8-426e-a508-47ecb4c7c310').get_cui(cui='C4304383')
pd.DataFrame(resp)
pd.DataFrame(pd.DataFrame(resp).result['inverseInheritedRelations'])