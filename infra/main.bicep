// Azure Bicep Template for FCA Compliance RAG Infrastructure
// Deploys: 
// 1. Azure OpenAI (with GPT-4o)
// 2. Azure AI Search (Basic/Standard)

@description('The name of the Azure OpenAI account')
param openAiName string = 'fca-rag-openai-${uniqueString(resourceGroup().id)}'

@description('The name of the Azure AI Search service')
param searchName string = 'fca-rag-search-${uniqueString(resourceGroup().id)}'

@description('The location for all resources')
param location string = resourceGroup().location

// OpenAI Service
resource openAi 'Microsoft.CognitiveServices/accounts@2023-05-01' = {
  name: openAiName
  location: location
  kind: 'OpenAI'
  sku: {
    name: 'S0'
  }
  properties: {
    customSubDomainName: openAiName
    publicNetworkAccess: 'Enabled' // In production bank, set to 'Disabled' + Private Endpoints
  }
}

// Deploy GPT-4o Model
resource gpt4o 'Microsoft.CognitiveServices/accounts/deployments@2023-05-01' = {
  parent: openAi
  name: 'gpt-4o'
  properties: {
    model: {
      format: 'OpenAI'
      name: 'gpt-4o'
      version: '2024-05-13'
    }
  }
  sku: {
    name: 'Standard'
    capacity: 10 // Units of 10k TPM
  }
}

// AI Search Service
resource search 'Microsoft.Search/searchServices@2023-11-01' = {
  name: searchName
  location: location
  sku: {
    name: 'basic' // Use 'standard' for production scale
  }
  properties: {
    replicaCount: 1
    partitionCount: 1
    hostingMode: 'default'
    publicNetworkAccess: 'Enabled'
  }
}

output openAiEndpoint string = openAi.properties.endpoint
output openAiName string = openAi.name
output searchEndpoint string = 'https://${searchName}.search.windows.net/'
output searchName string = search.name
