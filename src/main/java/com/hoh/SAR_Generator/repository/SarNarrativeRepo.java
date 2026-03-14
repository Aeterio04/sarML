package com.hoh.SAR_Generator.repository;

import com.hoh.SAR_Generator.model.entity.SarNarrativeVersion;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.UUID;

@Repository
public interface SarNarrativeRepo extends JpaRepository<SarNarrativeVersion, UUID> {
}