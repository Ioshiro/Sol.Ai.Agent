FROM mcr.microsoft.com/dotnet/sdk:8.0 AS build
WORKDIR /src

COPY SolAI.Pipecat.LLMService.csproj ./
RUN dotnet restore SolAI.Pipecat.LLMService.csproj

COPY . .
RUN dotnet publish SolAI.Pipecat.LLMService.csproj -c Release -o /app/publish /p:UseAppHost=false

FROM mcr.microsoft.com/dotnet/aspnet:8.0 AS runtime
WORKDIR /app
ENV ASPNETCORE_URLS=http://+:8080 \
    ASPNETCORE_ENVIRONMENT=Production
EXPOSE 8080
COPY --from=build /app/publish .
ENTRYPOINT ["dotnet", "SolAI.Pipecat.LLMService.dll"]
