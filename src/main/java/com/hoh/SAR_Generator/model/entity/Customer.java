package com.hoh.SAR_Generator.model.entity;

import jakarta.persistence.Entity;
import jakarta.persistence.Id;
import jakarta.persistence.Table;
import jakarta.persistence.Column;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDate;
import java.util.UUID;

@Entity
@Table(name = "customer")
@Data
@NoArgsConstructor
@AllArgsConstructor
public class Customer {

    @Id
    private UUID id;

    @Column(name = "customer_name", nullable = false)
    private String customerName;

    @Column(name = "customer_type")
    private String customerType;

    @Column(name = "risk_rating")
    private String riskRating;

    @Column(name = "country")
    private String country;

    @Column(name = "pep_flag")
    private Boolean pepFlag;

    @Column(name = "kyc_status")
    private String kycStatus;

    @Column(name = "dob")
    private LocalDate dob;

    @Column(name = "occupation")
    private String occupation;

    @Column(name = "created_at")
    private LocalDate createdAt;
}