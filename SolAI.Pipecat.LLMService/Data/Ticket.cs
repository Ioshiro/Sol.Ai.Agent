using System.ComponentModel.DataAnnotations;
using System.ComponentModel.DataAnnotations.Schema;

namespace SolAI.Pipecat.LLMService.Data;

[Table("tickets")]
public sealed class Ticket
{

    public const string STATO_APERTO = "Aperto";
    public const string STATO_CHIUSO = "Chiuso";

    [Key]
    [Column("id")]
    [DatabaseGenerated(DatabaseGeneratedOption.Identity)]
    public int Id { get; set; }

    [Required]
    [MaxLength(255)]
    [Column("Titolo")]
    public string Titolo { get; set; } = string.Empty;

    [Required]
    [Column("Descrizione", TypeName = "text")]
    public string Descrizione { get; set; } = string.Empty;

    [Required]
    [Column("DataCreazione", TypeName = "datetime")]
    public DateTime DataCreazione { get; set; }

    [Required]
    [MaxLength(255)]
    [Column("UtenteCreazione")]
    public string UtenteCreazione { get; set; } = string.Empty;

    [Required]
    [MaxLength(255)]
    [Column("Stato")]
    public string Stato { get; set; } = string.Empty;

    [Column("DataChiusura", TypeName = "datetime")]
    public DateTime? DataChiusura { get; set; }

    [MaxLength(255)]
    [Column("UtenteChiusura")]
    public string? UtenteChiusura { get; set; }

    [Column("Note", TypeName = "text")]
    public string? Note { get; set; }

}
