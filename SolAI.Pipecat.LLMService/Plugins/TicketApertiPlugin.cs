using System.ComponentModel;
using Microsoft.SemanticKernel;

namespace SolAI.Pipecat.LLMService.Plugins;

public sealed class TicketApertiPlugin
{
    private static readonly IReadOnlyDictionary<string, string> CustomerAliases =
        new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase)
        {
            ["olivetti srl"] = "Olivetti Srl",
            ["olivetti"] = "Olivetti Srl",
            ["microsoft srl"] = "Microsoft Srl",
            ["microsoft"] = "Microsoft Srl",
            ["apple srl"] = "Apple Srl",
            ["apple"] = "Apple Srl"
        };

    public sealed record TicketInfo(
        string Id,
        string Titolo,
        string Stato,
        string Priorita,
        string Assegnatario);

    public sealed record TicketCountResult(
        string CustomerName,
        bool CustomerFound,
        int Count);

    public sealed record OpenTicketsResult(
        string CustomerName,
        bool CustomerFound,
        int Count,
        IReadOnlyList<TicketInfo> Tickets);

    public static IReadOnlyCollection<string> KnownCustomerNames =>
        CustomerAliases.Values.Distinct(StringComparer.OrdinalIgnoreCase).ToArray();

    public static bool TryNormalizeCustomerName(string customerName, out string normalizedCustomerName)
    {
        var trimmedCustomerName = customerName.Trim();
        if (CustomerAliases.TryGetValue(trimmedCustomerName, out var canonicalCustomerName))
        {
            normalizedCustomerName = canonicalCustomerName;
            return true;
        }

        normalizedCustomerName = trimmedCustomerName;
        return false;
    }

    [KernelFunction("conta_ticket_aperti_cliente")]
    [Description("Conta i ticket aperti o in lavorazione del cliente richiesto. Usalo quando l'utente chiede quanti ticket aperti ha un cliente.")]
    public TicketCountResult GetTicketCount(
        [Description("Il nome del cliente. Accetta anche forme abbreviate come Olivetti, Microsoft o Apple.")] string customerName)
    {
        var normalizedCustomerName = NormalizeCustomerName(customerName);
        var tickets = GetTicketsByCustomer(normalizedCustomerName);
        return new TicketCountResult(normalizedCustomerName, tickets is not null, tickets?.Count ?? 0);
    }

    [KernelFunction("recupera_ticket_aperti_cliente")]
    [Description("Recupera l'elenco dei ticket aperti o in lavorazione del cliente richiesto con ID, titolo, stato, priorita' e assegnatario.")]
    public OpenTicketsResult GetTicketList(
        [Description("Il nome del cliente. Accetta anche forme abbreviate come Olivetti, Microsoft o Apple.")] string customerName)
    {
        var normalizedCustomerName = NormalizeCustomerName(customerName);
        var tickets = GetTicketsByCustomer(normalizedCustomerName);
        return new OpenTicketsResult(
            normalizedCustomerName,
            tickets is not null,
            tickets?.Count ?? 0,
            tickets ?? []);
    }

    [KernelFunction("recupera_dettaglio_ticket")]
    [Description("Recupera il dettaglio di un ticket specifico a partire dal suo ID, ad esempio TICKET-1022.")]
    public static string GetTicketDetails(
        [Description("L'identificativo del ticket, ad esempio TICKET-1022.")] string ticketId)
    {
        var normalizedTicketId = ticketId.Trim().ToUpperInvariant();
        if (normalizedTicketId == "TICKET-1022")
        {
            return "Il ticket TICKET-1022 è stato creato il 10/03/2026 alle 10:00 e ha lo stato Open.";
        }
        else if (normalizedTicketId == "TICKET-1023")
        {
            return "Il ticket TICKET-1023 è stato creato il 10/03/2026 alle 10:00 e ha lo stato Open.";
        }
        else if (normalizedTicketId == "TICKET-1024")
        {
            return "Il ticket TICKET-1024 è stato creato il 10/03/2026 alle 10:00 e ha lo stato Open.";
        }
        else if (normalizedTicketId == "TICKET-1025")
        {
            return "Il ticket TICKET-1025 è stato creato il 10/03/2026 alle 10:00 e ha lo stato In Progress.";
        }
        else if (normalizedTicketId == "TICKET-1026")
        {
            return "Il ticket TICKET-1026 è stato creato il 10/03/2026 alle 10:00 e ha lo stato Open.";
        }
        else if (normalizedTicketId == "TICKET-1027")
        {
            return "Il ticket TICKET-1027 è stato creato il 10/03/2026 alle 10:00 e ha lo stato Open.";
        }
        else if (normalizedTicketId == "TICKET-1028")
        {
            return "Il ticket TICKET-1028 è stato creato il 10/03/2026 alle 10:00 e ha lo stato In Progress.";
        }
        else if (normalizedTicketId == "TICKET-1029")
        {
            return "Il ticket TICKET-1029 è stato creato il 10/03/2026 alle 10:00 e ha lo stato Open.";
        }
        else if (normalizedTicketId == "TICKET-1030")
        {
            return "Il ticket TICKET-1030 è stato creato il 10/03/2026 alle 10:00 e ha lo stato Open.";
        }
        else if (normalizedTicketId == "TICKET-1031")
        {
            return "Il ticket TICKET-1031 è stato creato il 10/03/2026 alle 10:00 e ha lo stato In Progress.";
        }
        else if (normalizedTicketId == "TICKET-1032")
        {
            return "Il ticket TICKET-1032 è stato creato il 10/03/2026 alle 10:00 e ha lo stato Open.";
        }
        else if (normalizedTicketId == "TICKET-1033")
        {
            return "Il ticket TICKET-1033 è stato creato il 10/03/2026 alle 10:00 e ha lo stato Open.";
        }
        else if (normalizedTicketId == "TICKET-1034")
        {
            return "Il ticket TICKET-1034 è stato creato il 10/03/2026 alle 10:00 e ha lo stato In Progress.";
        }
        else
        {
            return "Il ticket non è riconosciuto.";
        }
    }

    private static List<TicketInfo>? GetTicketsByCustomer(string customerName)
    {
        var normalizedCustomerName = NormalizeCustomerName(customerName);
        if (normalizedCustomerName == "Olivetti Srl")
        {
            return
            [
                new("TICKET-1022", "Errore login utenti", "Open", "High", "Mario Rossi"),
                new("TICKET-1023", "Errore login utenti", "Open", "High", "Mario Rossi"),
                new("TICKET-1024", "Errore login utenti", "Open", "High", "Mario Rossi"),
                new("TICKET-1025", "Timeout integrazione API", "In Progress", "Medium", "Luca Bianchi"),
                new("TICKET-1026", "Allineamento dati CRM", "Open", "Low", "Giulia Verdi")
            ];
        }

        if (normalizedCustomerName == "Microsoft Srl")
        {
            return
            [
                new("TICKET-1027", "Permessi accesso dashboard", "Open", "High", "Alice Neri"),
                new("TICKET-1028", "Anomalia report export", "In Progress", "Medium", "Paolo Gialli"),
                new("TICKET-1029", "Verifica backup notturno", "Open", "Low", "Sara Blu")
            ];
        }

        if (normalizedCustomerName == "Apple Srl")
        {
            return
            [
                new("TICKET-1030", "Errore sincronizzazione ordini", "Open", "High", "Davide Viola"),
                new("TICKET-1031", "Notifiche email duplicate", "In Progress", "Medium", "Elena Rosa"),
                new("TICKET-1032", "Aggiornamento anagrafica cliente", "Open", "Low", "Franco Grigi"),
                new("TICKET-1033", "Errore sincronizzazione ordini", "Open", "High", "Davide Viola"),
                new("TICKET-1034", "Notifiche email duplicate", "In Progress", "Medium", "Elena Rosa")
            ];
        }

        return null;
    }

    private static string NormalizeCustomerName(string customerName)
    {
        return TryNormalizeCustomerName(customerName, out var normalizedCustomerName)
            ? normalizedCustomerName
            : customerName.Trim();
    }
}
