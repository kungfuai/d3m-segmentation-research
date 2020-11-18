import model_card_toolkit

# See https://github.com/tensorflow/model-card-toolkit for details on model card creation
model_card_output_path = "model_cards/your_model_name"
mct = model_card_toolkit.ModelCardToolkit(model_card_output_path)

# Initialize a new model card to populate
model_card = mct.scaffold_assets()
# model_card.model_details.name = ...
# model_card.model_details.citation = ...
# model_card.model_details.license = ...
# model_card.model_details.overview = ...
# model_card.model_details.owners.append(...)
# model_card.model_details.references.append(...)
# model_card.model_details.version.name = ...
# model_card.model_details.version.date = ...
# model_card.model_details.version.diff = ...
# model_card.considerations.users.append(...)
# model_card.considerations.use_cases.append(...)
# model_card.considerations.limitations.append(...)
# model_card.considerations.ethical_considerations.append(...)

# Write the model card data to a JSON file
mct.update_model_card_json(model_card)

# Return the model card document as an HTML page
html = mct.export_format()
