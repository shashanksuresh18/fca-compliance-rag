# deploy.ps1 — Helper script to deploy Azure RAG Infrastructure
# Usage: ./infra/deploy.ps1 -ResourceGroupName "fca-rag-rg" -Location "eastus"

param(
    [Parameter(Mandatory=$true)]
    [string]$ResourceGroupName,

    [string]$Location = "eastus"
)

Write-Host "--- Starting Azure RAG Infrastructure Deployment ---" -ForegroundColor Cyan

# 1. Create Resource Group
Write-Host "Creating Resource Group: $ResourceGroupName..."
az group create --name $ResourceGroupName --location $Location

# 2. Deploy Bicep
Write-Host "Deploying Bicep template (infra/main.bicep)... This may take 2-5 minutes."
$deployment = az deployment group create `
    --resource-group $ResourceGroupName `
    --template-file ./infra/main.bicep `
    --query properties.outputs `
    --output json | ConvertFrom-Json

# 3. Extract Outputs and Display in .env format
Write-Host "`n--- DEPLOYMENT COMPLETE ---" -ForegroundColor Green
Write-Host "Update your .env file with these values to switch to Azure mode:`n" -ForegroundColor Yellow

Write-Host "LLM_PROVIDER=AZURE"
Write-Host "AZURE_OPENAI_ENDPOINT=$($deployment.openAiEndpoint.value)"
Write-Host "AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4o"
Write-Host "AZURE_SEARCH_ENDPOINT=$($deployment.searchEndpoint.value)"
Write-Host "AZURE_SEARCH_INDEX_NAME=fca-compliance-index"

Write-Host "`nNote: You will also need to fetch your API keys from the Azure Portal:"
Write-Host "- OpenAI Key: 'Azure OpenAI > Keys and Endpoint'"
Write-Host "- Search Key: 'AI Search > Keys'"
