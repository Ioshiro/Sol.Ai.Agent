using System.ComponentModel.DataAnnotations;

namespace SolAI.Pipecat.LLMService.Options;

public sealed class LlmServiceOptions
{
    public const string SectionName = "LlmService";

    [Required]
    public string UpstreamEndpoint { get; init; } = "http://192.168.45.205:1234/v1";

    [Required]
    public string ApiKey { get; init; } = "lm-studio";

    public string? OrganizationId { get; init; }

    [Required]
    public string DefaultModel { get; init; } = "qwen3.5-4b";

    public List<string> AvailableModels { get; init; } = [];

    public string? ServerInstruction { get; init; } = "Sei un assistente vocale locale. " +
      "Rispondi in italiano, in modo chiaro e sintetico. " +
      "Per richieste su ticket, ticket aperti, dettaglio ticket, stato ticket, ordini, pratiche o altri dati operativi, devi usare prima i tool lato server quando disponibili. " +
      "Se manca un dato necessario per usare il tool, chiedi una sola informazione mancante invece di inventare la risposta. " +
      "Non citare nomi interni di plugin o componenti, a meno che l'utente non lo chieda esplicitamente. /no_think";

    [Required]
    public string StandardErrorMessage { get; init; } =
        "Mi dispiace, al momento il servizio linguistico non e' raggiungibile. Riprova tra qualche istante.";

    [Range(1, 20)]
    public int MaximumToolRounds { get; init; } = 5;
}
